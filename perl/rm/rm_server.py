import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from .math_verifier import extract_boxed_answer, compute_score
from .sglang_server import SGLangManager
from .config import RMConfig

logger = logging.getLogger("AllInOne-RM")

CONFIG = RMConfig()
sglang_manager = SGLangManager(CONFIG)
OUTPUT_DIR: Path | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup phase: boot SGLang and wait until it's ready.
    sglang_manager.start()
    await sglang_manager.wait_until_ready()
    
    yield
    
    # Shutdown phase: terminate SGLang safely.
    sglang_manager.stop()

app = FastAPI(title="All-in-One RM Server", lifespan=lifespan)


class RewardRequest(BaseModel):
    prompt: str
    response: str
    label: str
    metadata: dict | None = None


async def call_qwen_extractor(text: str) -> str:
    """Call the offline engine for answer extraction."""
    extraction_prompt = (
        "You are a math answer extractor. Extract the final answer. "
        "Output ONLY the answer itself (number/expression). "
        "If possible, use \\boxed{...}. Do NOT output any explanation.\n"
        "中文提示：只输出答案本身，不要输出多余文字。\n\n"
        f"Text:\n{text}"
    )
    messages = [{"role": "user", "content": extraction_prompt}]
    return await sglang_manager.async_generate_chat(messages)


async def clean_extracted_answer(text: str) -> str:
    """Ask the engine to normalize an extracted answer to a bare value."""
    cleanup_prompt = (
        "Normalize the following to ONLY the final answer. "
        "Output just the answer (number/expression), no extra words.\n"
        "中文提示：只输出答案本身。\n\n"
        f"Text:\n{text}"
    )
    messages = [{"role": "user", "content": cleanup_prompt}]
    return await sglang_manager.async_generate_chat(messages)


@app.post("/reward")
async def calculate_reward(req: RewardRequest):
    metadata = req.metadata or {}
    rm_type = metadata.get("rm_type", "math")

    if rm_type == "math":
        final_ans = None
        qwen_res = await call_qwen_extractor(req.response)
        print(qwen_res)
        if qwen_res:
            if "\\boxed" in qwen_res:
                final_ans = extract_boxed_answer(qwen_res)
            else:
                cleaned = await clean_extracted_answer(qwen_res)
                final_ans = cleaned.strip() if cleaned else None

        if not final_ans:
            score = 0.0
        else:
            score = compute_score(final_ans, req.label)

    else:
        logger.error(f"Unsupported RM type: {rm_type}")
        final_ans = None
        score = 0.0
        
    # Lightweight logging.
    logger.info(f"GT: {req.label[:20]}... | Extracted: {final_ans} | Score: {score}")

    # Persist request/response log per call.
    if OUTPUT_DIR is not None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = OUTPUT_DIR / f"{timestamp}_{uuid4().hex}.json"
        payload = {
            "timestamp": timestamp,
            "rm_type": rm_type,
            "prompt": req.prompt,
            "response": req.response,
            "label": req.label,
            "metadata": metadata,
            "extracted": final_ans,
            "score": score,
        }
        log_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    
    return score

@app.get("/health")
def health():
    return {"status": "running"}

def run_rm_server():
    parser = argparse.ArgumentParser(description="All-in-One RM Server")
    parser.add_argument("--model-path", default=RMConfig.model_path)
    parser.add_argument("--tp-size", type=int, default=RMConfig.sglang_tp_size)
    parser.add_argument("--dp-size", type=int, default=RMConfig.sglang_dp_size)
    parser.add_argument("--rm-server-port", type=int, default=RMConfig.rm_server_port)
    parser.add_argument("--rm-host", default="0.0.0.0")
    parser.add_argument("--output-dir", default="rm_logs")
    args = parser.parse_args()

    # Initialize the config and manager.
    global CONFIG, sglang_manager, OUTPUT_DIR
    CONFIG = RMConfig(
        model_path=args.model_path,
        sglang_tp_size=args.tp_size,
        sglang_dp_size=args.dp_size,
        rm_server_port=args.rm_server_port,
    )
    sglang_manager = SGLangManager(CONFIG)
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Start the main service.
    print("🔥 Starting All-in-One RM service...")
    print(f"👉 HTTP port: {CONFIG.rm_server_port}")
    
    uvicorn.run(app, host=args.rm_host, port=CONFIG.rm_server_port)

if __name__ == "__main__":
    run_rm_server()