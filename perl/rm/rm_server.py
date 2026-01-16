import argparse
import logging
from contextlib import asynccontextmanager

import aiohttp
import uvicorn
from fastapi import FastAPI

from .math_verifier import extract_answer, extract_boxed_answer, compute_score
from .sglang_server import SGLangManager
from .config import RMConfig

logger = logging.getLogger("AllInOne-RM")

CONFIG = RMConfig()
sglang_manager = SGLangManager(CONFIG)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup phase: boot SGLang and wait until it's ready.
    sglang_manager.start()
    await sglang_manager.wait_until_ready()
    
    yield
    
    # Shutdown phase: terminate SGLang safely.
    sglang_manager.stop()

app = FastAPI(title="All-in-One RM Server", lifespan=lifespan)

@app.post("/reward")
async def calculate_reward_with_rm_server(args, sample):
    
    prompt = sample.prompt  
    response = sample.response  
    label = sample.label 
    metadata = sample.metadata # must contains a rm_type

    rm_type = metadata.get("rm_type", "math")

    if rm_type == "math":

        qwen_res = await call_qwen_extractor(req.response)
        final_ans = extract_boxed_answer(qwen_res)
        score = compute_score(final_ans, req.label)

    else:
        
        logger.error(f"Unsupported RM type: {rm_type}")
        score = 0.0
        
    # Lightweight logging.
    logger.info(f"GT: {req.label[:20]}... | Extracted: {final_ans} | Score: {score}")
    
    return score

@app.get("/health")
def health():
    return {"status": "running"}

def run_rm_server():
    parser = argparse.ArgumentParser(description="All-in-One RM Server")
    parser.add_argument("--model-path", default=RMConfig.model_path)
    parser.add_argument("--sglang-port", type=int, default=RMConfig.sglang_port)
    parser.add_argument("--sglang-host", default=RMConfig.sglang_host)
    parser.add_argument("--tp-size", type=int, default=RMConfig.sglang_tp_size)
    parser.add_argument("--dp-size", type=int, default=RMConfig.sglang_dp_size)
    parser.add_argument("--rm-server-port", type=int, default=RMConfig.rm_server_port)
    parser.add_argument("--rm-host", default="0.0.0.0")
    args = parser.parse_args()

    # Initialize the config and manager.
    global CONFIG, sglang_manager
    CONFIG = RMConfig(
        model_path=args.model_path,
        sglang_port=args.sglang_port,
        sglang_host=args.sglang_host,
        sglang_tp_size=args.tp_size,
        sglang_dp_size=args.dp_size,
        rm_server_port=args.rm_server_port,
    )
    sglang_manager = SGLangManager(CONFIG)

    # Start the main service.
    print("🔥 Starting All-in-One RM service...")
    print(f"👉 HTTP port: {CONFIG.rm_server_port}")
    print(f"👉 SGLang port: {CONFIG.sglang_port}")
    
    uvicorn.run(app, host=args.rm_host, port=CONFIG.rm_server_port)

if __name__ == "__main__":
    run_rm_server()