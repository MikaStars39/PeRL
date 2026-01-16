import logging
from typing import Any, Dict, List, Optional

import sglang as sgl
from transformers import AutoTokenizer

logger = logging.getLogger("AllInOne-RM")


class SGLangManager:
    """Lifecycle manager for the SGLang offline engine."""

    def __init__(self, config):
        self.config = config
        self.engine = None
        self.tokenizer = None

    def start(self):
        """Start the SGLang offline engine."""
        logger.info(f"🚀 Starting SGLang offline engine (Model: {self.config.model_path})...")
        self.engine = sgl.Engine(
            model_path=self.config.model_path,
            tp_size=self.config.sglang_tp_size,
            dp_size=self.config.sglang_dp_size,
            trust_remote_code=self.config.sglang_trust_remote_code,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, trust_remote_code=self.config.sglang_trust_remote_code
        )

    async def wait_until_ready(self):
        """No-op for offline engine (kept for lifecycle compatibility)."""
        return

    def stop(self):
        """Shutdown the SGLang offline engine."""
        if self.engine:
            logger.info("🛑 Shutting down SGLang offline engine...")
            try:
                self.engine.shutdown()
            except Exception as e:
                logger.warning(f"Shutdown issue (may already be closed): {e}")
            logger.info("👋 SGLang offline engine stopped.")

    def generate(self, prompt: str, sampling_params: Optional[Dict[str, Any]] = None) -> str:
        """Run a single-prompt generation on the offline engine."""
        if not self.engine:
            raise RuntimeError("SGLang engine is not initialized.")
        params = sampling_params or {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 128}
        outputs = self.engine.generate([prompt], params)
        return outputs[0].get("text", "").strip()

    async def async_generate(self, prompt: str, sampling_params: Optional[Dict[str, Any]] = None) -> str:
        """Run a single-prompt async generation on the offline engine."""
        if not self.engine:
            raise RuntimeError("SGLang engine is not initialized.")
        params = sampling_params or {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 128}
        outputs = await self.engine.async_generate([prompt], params)
        return outputs[0].get("text", "").strip()

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        if not self.tokenizer:
            raise RuntimeError("Tokenizer is not initialized.")
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            logger.info(f"[template] apply_chat_template failed; fallback to raw prompt. Error: {e}")
            return messages[-1]["content"]

    async def async_generate_chat(
        self, messages: List[Dict[str, str]], sampling_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Run async generation with chat template applied."""
        prompt = self._apply_chat_template(messages)
        return await self.async_generate(prompt, sampling_params)
