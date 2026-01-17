import logging
from typing import Any, Dict, List, Optional

import sglang as sgl
from transformers import AutoTokenizer

from .config import RMConfig

logger = logging.getLogger("RM")


# ------------ SGLang Offline Engine Manager ------------

class SGLangManager:
    """Manages SGLang offline engine lifecycle and generation."""

    def __init__(self, config: RMConfig):
        self.config = config
        self.engine: Optional[sgl.Engine] = None
        self.tokenizer = None

    # ------------ Lifecycle ------------

    def start(self):
        """Initialize SGLang engine and tokenizer."""
        logger.info(f"Starting SGLang engine: {self.config.model_path}")
        self.engine = sgl.Engine(
            model_path=self.config.model_path,
            tp_size=self.config.tp_size,
            dp_size=self.config.dp_size,
            trust_remote_code=self.config.trust_remote_code,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=self.config.trust_remote_code,
        )

    def stop(self):
        """Shutdown SGLang engine."""
        if self.engine:
            logger.info("Stopping SGLang engine...")
            try:
                self.engine.shutdown()
            except Exception as e:
                logger.warning(f"Shutdown warning: {e}")

    async def wait_until_ready(self):
        """No-op for offline engine (kept for interface consistency)."""
        pass

    # ------------ Generation ------------

    async def generate(
        self,
        prompt: str,
        sampling_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Async generate from raw prompt."""
        if not self.engine:
            raise RuntimeError("Engine not initialized")
        params = sampling_params or self.config.sampling_params
        outputs = await self.engine.async_generate([prompt], params)
        return outputs[0].get("text", "").strip()

    async def chat(
        self,
        messages: List[Dict[str, str]],
        sampling_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Async generate with chat template applied."""
        prompt = self._apply_chat_template(messages)
        return await self.generate(prompt, sampling_params)

    # ------------ Internal ------------

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Apply tokenizer chat template to messages."""
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not initialized")
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback: return last message content
            return messages[-1]["content"]
