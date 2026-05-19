import os
import gc
import logging
import asyncio
import warnings
import sglang as sgl
from typing import Optional, Dict, Any
from transformers import AutoTokenizer

# Configure Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SGLangBase")


def _patch_resource_tracker():
    """
    Patch multiprocessing.resource_tracker to suppress harmless KeyError warnings.
    
    This error occurs when SGLang's child processes clean up shared memory resources
    before the main process's resource_tracker tries to remove them from its cache.
    The error is harmless since the resources are already properly cleaned up.
    """
    try:
        from multiprocessing import resource_tracker
        
        # Patch the unregister function
        _original_unregister = resource_tracker.unregister
        def _safe_unregister(name, rtype):
            try:
                return _original_unregister(name, rtype)
            except (KeyError, FileNotFoundError):
                pass  # Resource already cleaned up
        resource_tracker.unregister = _safe_unregister
        
        # Patch the ResourceTracker class methods if available
        if hasattr(resource_tracker, 'ResourceTracker'):
            _original_unlink = getattr(resource_tracker.ResourceTracker, '_unlink', None)
            if _original_unlink:
                def _safe_unlink(self, name, rtype):
                    try:
                        return _original_unlink(self, name, rtype)
                    except (KeyError, FileNotFoundError):
                        pass
                resource_tracker.ResourceTracker._unlink = _safe_unlink
                
    except Exception:
        pass  # Silently ignore if patching fails


# Apply patches at module load time
_patch_resource_tracker()

# Suppress related warnings
warnings.filterwarnings("ignore", message=".*resource_tracker.*")

class BaseSGLangEngine:
    """
    Base engine handling SGLang lifecycle, configuration, and safe generation.
    """

    def __init__(
        self,
        model_path: str,
        dp_size: int = 1,
        tp_size: int = 1,
        mem_fraction_static: float = 0.90,
        enable_radix_cache: bool = False,
        # Speculative Decoding Config
        speculative_algorithm: Optional[str] = None,
        speculative_draft_model_path: Optional[str] = None,
        speculative_num_steps: Optional[int] = None,
        speculative_eagle_topk: Optional[int] = None,
        speculative_num_draft_tokens: Optional[int] = None,
    ):
        # ------------------------- Configuration ------------------------
        self.model_path = model_path
        
        # Engine arguments
        self.engine_args = {
            "model_path": model_path,
            "dp_size": dp_size,
            "tp_size": tp_size,
            "mem_fraction_static": mem_fraction_static,
            "disable_radix_cache": not enable_radix_cache,
            "trust_remote_code": True,
            "log_level": "error",  # Keep quiet by default
        }

        # Speculative arguments
        self.spec_args = {
            "speculative_algorithm": speculative_algorithm,
            "speculative_draft_model_path": speculative_draft_model_path,
            "speculative_num_steps": speculative_num_steps,
            "speculative_eagle_topk": speculative_eagle_topk,
            "speculative_num_draft_tokens": speculative_num_draft_tokens,
        }

        # ------------------------- Runtime State ------------------------
        self.llm: Optional[sgl.Engine] = None
        self.tokenizer = None

    # ------------------------- Lifecycle Management ------------------------

    async def __aenter__(self):
        """Initializes the Engine and Tokenizer."""
        os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
        logger.info(f"Initializing Engine (TP={self.engine_args['tp_size']})...")

        # 1. Start SGLang Engine
        self.llm = sgl.Engine(**self.engine_args, **self.spec_args)
        
        # 2. Load Tokenizer (Useful for chat templates in both modes)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"Tokenizer load warning: {e}")

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Graceful shutdown with proper resource cleanup."""
        if self.llm:
            logger.info("Shutting down Engine...")
            try:
                self.llm.shutdown()
            except Exception as e:
                logger.warning(f"Engine shutdown warning: {e}")
            
            # Allow time for subprocess cleanup to prevent resource_tracker KeyError
            await asyncio.sleep(1.0)
            
            # Force garbage collection to release shared memory references
            gc.collect()
            
            self.llm = None
            
        if exc_val:
            logger.error(f"Engine exited with error: {exc_val}")

    # ------------------------- Core Helpers ------------------------

    async def _generate_safe(self, prompt: str, params: dict) -> Dict[str, Any]:
        """
        Wraps generation with a fallback mechanism for incompatible params.
        Shared by both Online and Offline modes.
        """
        try:
            return await self.llm.async_generate(prompt, params)
        except Exception as e:
            # Fallback: Drop keys that might cause issues
            drop_keys = {
                "stop", "stop_token_ids", "repetition_penalty", 
                "frequency_penalty", "presence_penalty", "min_new_tokens"
            }
            filtered = {k: v for k, v in params.items() if k not in drop_keys}
            
            if filtered == params:
                raise e 
            
            logger.warning(f"Retrying without optional params due to: {e}")
            return await self.llm.async_generate(prompt, filtered)