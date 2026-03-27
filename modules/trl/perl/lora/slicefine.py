# slicefine_peft.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.config import PeftConfig
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_adapters_to_merge

@dataclass
class SliceFineConfig(PeftConfig):
    r: int = field(default=8, metadata={"help": "Rank (width) of the trainable slice."})
    slice_mode: str = field(default="column", metadata={"help": "'column' or 'row'."})
    slice_position: int = field(default=0, metadata={"help": "Starting index."})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None, metadata={"help": "Modules to replace."}
    )
    bias: str = field(default="none")
    modules_to_save: Optional[List[str]] = field(default=None)

    def __post_init__(self):
        self.peft_type = "SLICEFINE"
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )

class SliceFineLayer(BaseTunerLayer):
    # Only trainable parts go here to prevent PEFT from messing with frozen parts
    adapter_layer_names = ("slicefine_adapters",)
    other_param_names = ("slice_r", "slice_mode", "slice_position")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.slice_r = {}
        self.slice_mode = {}
        self.slice_position = {}
        
        # Frozen parts: stored as ModuleDict of Linear layers
        self.slicefine_A_layers = nn.ModuleDict({})
        self.slicefine_B_layers = nn.ModuleDict({})
        
        # Trainable part
        self.slicefine_adapters = nn.ModuleDict({})
        
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        if isinstance(base_layer, nn.Linear):
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

    def update_layer(self, adapter_name, r, slice_mode, slice_position, **kwargs):
        self.slice_r[adapter_name] = r
        self.slice_mode[adapter_name] = slice_mode
        self.slice_position[adapter_name] = slice_position

        base_layer = self.get_base_layer()
        weight = base_layer.weight
        dtype = weight.dtype
        device = weight.device
        
        # Helper to create a frozen linear layer from a weight slice
        def create_frozen_linear(weight_slice, fan_in, fan_out):
            layer = nn.Linear(fan_in, fan_out, bias=False)
            layer.weight.data = weight_slice
            layer.weight.requires_grad = False
            layer.to(device=device, dtype=dtype)
            return layer

        # Helper to create the trainable linear layer
        def create_trainable_linear(weight_slice, fan_in, fan_out):
            layer = nn.Linear(fan_in, fan_out, bias=False)
            layer.weight.data = weight_slice
            layer.weight.requires_grad = True # Explicitly True
            layer.to(device=device, dtype=dtype)
            return layer

        if slice_mode == "column":
            total_cols = weight.shape[1]
            if slice_position + r > total_cols:
                slice_position = max(0, total_cols - r)
            
            # --- Column Slicing Logic ---
            # Part A (Left)
            width_a = slice_position
            if width_a > 0:
                self.slicefine_A_layers[adapter_name] = create_frozen_linear(
                    weight[:, :width_a].detach().clone(), width_a, self.out_features
                )
            
            # Part T (Middle - Trainable)
            self.slicefine_adapters[adapter_name] = create_trainable_linear(
                weight[:, slice_position : slice_position + r].detach().clone(), r, self.out_features
            )
            
            # Part B (Right)
            width_b = total_cols - (slice_position + r)
            if width_b > 0:
                self.slicefine_B_layers[adapter_name] = create_frozen_linear(
                    weight[:, slice_position + r :].detach().clone(), width_b, self.out_features
                )
            
        elif slice_mode == "row":
            total_rows = weight.shape[0]
            if slice_position + r > total_rows:
                slice_position = max(0, total_rows - r)

            # --- Row Slicing Logic ---
            
            # Part A (Top)
            height_a = slice_position
            if height_a > 0:
                self.slicefine_A_layers[adapter_name] = create_frozen_linear(
                    weight[:height_a, :].detach().clone(), self.in_features, height_a
                )

            # Part T (Middle - Trainable)
            self.slicefine_adapters[adapter_name] = create_trainable_linear(
                weight[slice_position : slice_position + r, :].detach().clone(), self.in_features, r
            )

            # Part B (Bottom)
            height_b = total_rows - (slice_position + r)
            if height_b > 0:
                self.slicefine_B_layers[adapter_name] = create_frozen_linear(
                    weight[slice_position + r :, :].detach().clone(), self.in_features, height_b
                )
        
        self.slice_position[adapter_name] = slice_position
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

class SliceFineLinear(nn.Module, SliceFineLayer):
    def __init__(self, base_layer, adapter_name: str, r: int = 8, slice_mode: str = "column", slice_position: int = 0, **kwargs):
        super().__init__()
        SliceFineLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, slice_mode, slice_position, **kwargs)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.slicefine_adapters.keys():
                base_layer = self.get_base_layer()
                mode = self.slice_mode[active_adapter]
                
                # Get parts
                part_T = self.slicefine_adapters[active_adapter].weight
                
                # === FIX: ModuleDict has no .get() ===
                part_A = None
                if active_adapter in self.slicefine_A_layers:
                    part_A = self.slicefine_A_layers[active_adapter].weight
                
                part_B = None
                if active_adapter in self.slicefine_B_layers:
                    part_B = self.slicefine_B_layers[active_adapter].weight
                
                # Concatenate
                tensors_to_cat = []
                if part_A is not None: tensors_to_cat.append(part_A)
                tensors_to_cat.append(part_T)
                if part_B is not None: tensors_to_cat.append(part_B)
                
                dim = 1 if mode == "column" else 0
                merged_weight = torch.cat(tensors_to_cat, dim=dim)
                
                base_layer.weight.data = merged_weight.to(base_layer.weight.dtype)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            return
        self.merged_adapters.clear()

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters or self.merged:
            return self.base_layer(x, *args, **kwargs)

        for active_adapter in self.active_adapters:
            if active_adapter not in self.slicefine_adapters.keys():
                continue
            
            mode = self.slice_mode[active_adapter]
            adapter_layer = self.slicefine_adapters[active_adapter]
            
            # === FIX: ModuleDict has no .get() ===
            layer_A = None
            if active_adapter in self.slicefine_A_layers:
                layer_A = self.slicefine_A_layers[active_adapter]
                
            layer_B = None
            if active_adapter in self.slicefine_B_layers:
                layer_B = self.slicefine_B_layers[active_adapter]
            
            dtype = adapter_layer.weight.dtype
            x = x.to(dtype)

            if mode == "column":
                # A (frozen) | T (trainable) | B (frozen)
                components = []
                current_idx = 0
                
                if layer_A is not None:
                    in_a = layer_A.weight.shape[1]
                    components.append(F.linear(x[..., :in_a], layer_A.weight))
                    current_idx += in_a
                
                in_t = adapter_layer.weight.shape[1]
                components.append(adapter_layer(x[..., current_idx : current_idx + in_t]))
                current_idx += in_t
                
                if layer_B is not None:
                    components.append(F.linear(x[..., current_idx:], layer_B.weight))
                
                out = sum(components)

            else: # row
                # Stack vertically
                components = []
                if layer_A is not None:
                    components.append(F.linear(x, layer_A.weight))
                
                components.append(adapter_layer(x))
                
                if layer_B is not None:
                    components.append(F.linear(x, layer_B.weight))
                
                out = torch.cat(components, dim=-1)
            
            if self.base_layer.bias is not None:
                out += self.base_layer.bias
            return out
            
        return self.base_layer(x, *args, **kwargs)

class SliceFineModel(BaseTuner):
    prefix: str = "slicefine_"
    tuner_layer_cls = SliceFineLayer

    def __init__(self, model, config, adapter_name="default"):
        super().__init__(model, config, adapter_name)
        self._ensure_only_adapter_trainable(adapter_name)

    def _ensure_only_adapter_trainable(self, adapter_name):
        trainable_count = 0
        frozen_count = 0
        for name, param in self.model.named_parameters():
            if adapter_name in name:
                if "slicefine_adapters" in name:
                    param.requires_grad = True
                    trainable_count += 1
                elif "slicefine_A_layers" in name or "slicefine_B_layers" in name:
                    param.requires_grad = False
                    frozen_count += 1
        print(f"[SliceFine] Safety Check: {trainable_count} params set to trainable, {frozen_count} params frozen.")

    def merge_adapter(self, adapter_names: Optional[List[str]] = None) -> None:
        for module in self.model.modules():
            if isinstance(module, SliceFineLayer):
                module.merge(adapter_names=adapter_names)

    def unmerge_adapter(self) -> None:
        for module in self.model.modules():
            if isinstance(module, SliceFineLayer):
                module.unmerge()

    def _create_and_replace(self, slicefine_config, adapter_name, target, target_name, parent, current_key, **optional_kwargs):
        if isinstance(target, SliceFineLayer):
            target.update_layer(adapter_name, r=slicefine_config.r, slice_mode=slicefine_config.slice_mode, slice_position=slicefine_config.slice_position)
        else:
            kwargs = {"r": slicefine_config.r, "slice_mode": slicefine_config.slice_mode, "slice_position": slicefine_config.slice_position}
            new_module = SliceFineLinear(target, adapter_name, **kwargs)
            self._replace_module(parent, target_name, new_module, target)
            
def register_slicefine_method():
    import peft.mapping
    SLICEFINE_TYPE = "SLICEFINE"
    peft.mapping.PEFT_TYPE_TO_CONFIG_MAPPING[SLICEFINE_TYPE] = SliceFineConfig
    peft.mapping.PEFT_TYPE_TO_TUNER_MAPPING[SLICEFINE_TYPE] = SliceFineModel
