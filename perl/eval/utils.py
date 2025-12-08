# logging_utils.py
import logging
import sys
import time
import torch
import gc
import argparse

from typing import Any, Dict
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path


# ------------------ model utils ---------------------


def resolve_torch_dtype(dtype: Any) -> Any:
    """
    Parse dtype string to torch.dtype, supporting auto/common aliases,
    compatible with older Transformers lacking get_torch_dtype.
    """
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        normalized = dtype.lower()
        if normalized == "auto":
            return None
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if normalized in mapping:
            return mapping[normalized]
    raise ValueError(f"Unsupported dtype: {dtype}")


def merge_model_if_needed(
    args: argparse.Namespace, result_dir: Path, logger: logging.Logger
) -> Path:
    if not args.adapter:
        logger.info("No adapter provided, using base model directly: %s", args.model)
        return Path(args.model)

    output_dir = result_dir / "model"
    if output_dir.exists() and any(output_dir.iterdir()):
        logger.info("Detected existing merged model directory, reusing: %s", output_dir)
        return output_dir

    adapter_path = Path(args.adapter).resolve()
    if not adapter_path.exists():
        raise ValueError(f"Adapter path does not exist: {adapter_path}")

    # Check for critical files to prevent peft from mistaking it for a huggingface repo id
    adapter_config_path = adapter_path / "adapter_config.json"
    if not adapter_config_path.exists():
        raise ValueError(f"adapter_config.json not found in {adapter_path}")

    torch_dtype = resolve_torch_dtype(args.dtype)
    logger.info("Loading base model: %s", args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="cuda",
        trust_remote_code=args.trust_remote_code,
    )
    logger.info("Loading tokenizer: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    logger.info("Loading LoRA/PEFT adapter: %s", adapter_path)
    # Explicitly cast to str
    model = PeftModel.from_pretrained(model, str(adapter_path), device_map="cuda")
    logger.info("Executing merge_and_unload, writing LoRA weights into base model.")
    model = model.merge_and_unload()

    logger.info("Saving merged model to: %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    # Actively release memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return output_dir


# ------------------ logging utils ---------------------


class StreamToLogger:
    """Redirect stdout/stderr to logger to ensure output is recorded in both file and console."""

    def __init__(self, logger: logging.Logger, level: int) -> None:
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, buffer: str) -> None:
        self._buffer += buffer
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self.logger.log(self.level, line)

    def flush(self) -> None:
        if self._buffer:
            self.logger.log(self.level, self._buffer)
            self._buffer = ""


def setup_logging(result_dir: Path) -> logging.Logger:
    result_dir.mkdir(parents=True, exist_ok=True)
    latest_log_path = result_dir / "latest_run.log"
    log_path = result_dir / "logs" / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    latest_file_handler = logging.FileHandler(
        latest_log_path, mode="w", encoding="utf-8"
    )
    latest_file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.__stdout__)
    console_handler.setFormatter(formatter)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(latest_file_handler)
    logging.root.addHandler(console_handler)

    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    stdout_logger.propagate = True
    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    stderr_logger.propagate = True
    sys.stdout = StreamToLogger(stdout_logger, logging.INFO)
    sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)

    return logging.getLogger("eval_all")


class StageContext:
    """Stage logging context to mark start/end and failure scenarios."""

    def __init__(
        self,
        logger: logging.Logger,
        stage_id: int | str,
        name: str,
        emoji_start: str = "🚀",
        emoji_end: str = "🏁",
        emoji_fail: str = "💥",
    ) -> None:
        self.logger = logger
        self.stage_id = str(stage_id)
        self.name = name
        self.emoji_start = emoji_start
        self.emoji_end = emoji_end
        self.emoji_fail = emoji_fail

    def __enter__(self) -> "StageContext":
        self.logger.info(
            "%s Stage %s started: %s", self.emoji_start, self.stage_id, self.name
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        if exc_type is None:
            self.logger.info(
                "%s Stage %s ended: %s", self.emoji_end, self.stage_id, self.name
            )
        else:
            self.logger.error(
                "%s Stage %s failed: %s, Error: %s",
                self.emoji_fail,
                self.stage_id,
                self.name,
                exc,
            )
