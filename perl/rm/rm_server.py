import argparse
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import anyio
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import RMConfig
from .math_verifier import extract_boxed_answer, compute_score
from .sglang_server import SGLangManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("RM")


# ------------ Global State ------------

config: RMConfig = None
engine: SGLangManager = None
semaphore: asyncio.Semaphore = None


# ------------ Request Schema ------------

class RewardRequest(BaseModel):
    prompt: str
    response: str
    label: str
    metadata: dict | None = None


# ------------ Answer Extraction ------------

EXTRACTION_PROMPT = """Reference Format (Ground Truth): {label}

Model Reasoning Process:
{response}

Task: Extract the final answer from the reasoning process above.
Instructions:
1. Follow the style/format of the Reference Format.
2. DO NOT correct any mistakes. Extract what the model actually concluded, even if wrong.
3. DO NOT simplify the answer. If the model concludes with an equation (e.g., 'x = 0'), extract the FULL equation: \\boxed{{x = 0}}, NOT just \\boxed{{0}}.
4. You can do short analysis. Your final response must end with \\boxed{{answer}} format."""


async def extract_answer(response: str, label: str) -> str | None:
    """Extract answer from model response using LLM."""
    prompt = EXTRACTION_PROMPT.format(response=response, label=label)
    result = await engine.chat([{"role": "user", "content": prompt}])
    return extract_boxed_answer(result) if result else None


# ------------ Reward Calculation ------------

async def calculate_math_reward(response: str, label: str) -> tuple[float, str | None]:
    """Calculate reward for math task. Returns (score, extracted_answer)."""
    # Must contain exactly one </think> tag
    if "</think>" not in response or response.count("</think>") != 1:
        return 0.0, None

    # Extract content after </think>
    answer_part = response.split("</think>")[1].strip()
    if not answer_part:
        return 0.0, None

    # Extract and verify answer
    with anyio.fail_after(config.timeout):
        extracted = await extract_answer(answer_part, label)

    if not extracted:
        logger.warning("Failed to extract boxed answer")
        return 0.0, None

    score = compute_score(extracted, label)
    return score, extracted


# ------------ Logging ------------

async def save_log(req: RewardRequest, extracted: str | None, score: float, rm_type: str):
    """Async save request log to file."""
    if not config.output_dir:
        return
    try:
        output_path = Path(config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_file = output_path / f"{timestamp}_{uuid4().hex[:8]}.json"
        
        payload = {
            "timestamp": timestamp,
            "rm_type": rm_type,
            "prompt": req.prompt,
            "response": req.response,
            "label": req.label,
            "metadata": req.metadata,
            "extracted": extracted,
            "score": score,
        }
        content = json.dumps(payload, ensure_ascii=False)
        await anyio.to_thread.run_sync(log_file.write_text, content, "utf-8")
    except Exception as e:
        logger.error(f"Failed to save log: {e}")


# ------------ FastAPI App ------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage SGLang engine lifecycle."""
    engine.start()
    await engine.wait_until_ready()
    yield
    engine.stop()


app = FastAPI(title="RM Server", lifespan=lifespan)


@app.post("/reward")
async def reward_endpoint(req: RewardRequest):
    """Calculate reward for a given response."""
    async with semaphore:
        try:
            metadata = req.metadata or {}
            rm_type = metadata.get("rm_type", "math")
            score, extracted = 0.0, None

            if rm_type == "math":
                score, extracted = await calculate_math_reward(req.response, req.label)
            else:
                logger.error(f"Unsupported rm_type: {rm_type}")

            # Log
            preview = req.label[:20] if req.label else ""
            logger.info(f"GT: {preview}... | Extracted: {extracted} | Score: {score}")
            asyncio.create_task(save_log(req, extracted, score, rm_type))

            return score

        except TimeoutError:
            logger.error("Request timed out")
            return 0.0
        except Exception as e:
            logger.error(f"Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


# ------------ CLI Entry ------------

def parse_args() -> RMConfig:
    """Parse CLI args and return config."""
    parser = argparse.ArgumentParser(description="RM Server")
    parser.add_argument("--model-path", default=RMConfig.model_path)
    parser.add_argument("--tp-size", type=int, default=RMConfig.tp_size)
    parser.add_argument("--dp-size", type=int, default=RMConfig.dp_size)
    parser.add_argument("--temperature", type=float, default=RMConfig.temperature)
    parser.add_argument("--top-p", type=float, default=RMConfig.top_p)
    parser.add_argument("--max-new-tokens", type=int, default=RMConfig.max_new_tokens)
    parser.add_argument("--host", default=RMConfig.host)
    parser.add_argument("--port", type=int, default=RMConfig.port)
    parser.add_argument("--max-concurrent", type=int, default=RMConfig.max_concurrent)
    parser.add_argument("--timeout", type=int, default=RMConfig.timeout)
    parser.add_argument("--output-dir", default=RMConfig.output_dir)
    args = parser.parse_args()

    return RMConfig(
        model_path=args.model_path,
        tp_size=args.tp_size,
        dp_size=args.dp_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        host=args.host,
        port=args.port,
        max_concurrent=args.max_concurrent,
        timeout=args.timeout,
        output_dir=args.output_dir,
    )


def run_rm_server():
    """Main entry point."""
    global config, engine, semaphore

    config = parse_args()
    engine = SGLangManager(config)
    semaphore = asyncio.Semaphore(config.max_concurrent)

    print(f"🔥 Starting RM Server on {config.host}:{config.port}")
    print(f"   Model: {config.model_path}")
    print(f"   TP={config.tp_size}, DP={config.dp_size}")
    print(f"   MaxConcurrent={config.max_concurrent}, Timeout={config.timeout}s")

    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    run_rm_server()
