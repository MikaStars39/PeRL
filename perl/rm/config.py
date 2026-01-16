from dataclasses import dataclass
from pydantic import BaseModel

@dataclass(frozen=True)
class RMConfig:
    """Centralized runtime settings for RM and SGLang."""

    model_path: str = "Qwen/Qwen2.5-Math-7B-Instruct"
    sglang_port: int = 30000
    sglang_host: str = "0.0.0.0"
    sglang_tp_size: int = 1
    sglang_dp_size: int = 1
    rm_server_port: int = 8000

    @property
    def sglang_url(self) -> str:
        return f"http://localhost:{self.sglang_port}/v1/chat/completions"
