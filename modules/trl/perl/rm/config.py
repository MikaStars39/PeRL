from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class RMConfig:
    """Centralized configuration for RM Server and SGLang Engine."""

    # ------------ Model Settings ------------
    model_path: str = "Qwen/Qwen2.5-Math-7B-Instruct"
    trust_remote_code: bool = True

    # ------------ SGLang Engine ------------
    tp_size: int = 1
    dp_size: int = 1

    # ------------ Generation Settings ------------
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 256

    # ------------ Server Settings ------------
    host: str = "0.0.0.0"
    port: int = 8000

    # ------------ Concurrency & Timeout ------------
    max_concurrent: int = 16
    timeout: int = 30

    # ------------ Logging ------------
    output_dir: str = "rm_logs"

    @property
    def sampling_params(self) -> Dict[str, Any]:
        """Return sampling params dict for SGLang."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
        }
