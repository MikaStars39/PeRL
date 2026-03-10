import megatron.bridge
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained("/mnt/llm-train/users/explore-train/qingyu/.cache/DeepSeek-R1-Distill-Qwen-1.5B", trust_remote_code=True)
print("megatron.bridge file:", megatron.bridge.__file__)
print("bridge type:", type(bridge))
print("has export_adapter_weights:", hasattr(bridge, "export_adapter_weights"))