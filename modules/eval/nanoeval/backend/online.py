import asyncio
import aiohttp
import json
import time
import os
import sys
import logging
import argparse
import signal
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field

# Try to import tqdm
try:
    from tqdm.asyncio import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable

# ------------------------- Configuration ------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr
)
logger = logging.getLogger("OnlineInference")


def _extract_message_content_and_reasoning(response: Dict[str, Any]) -> tuple[str, Optional[str]]:
    choices = response.get("choices") or []
    if not choices:
        return "", None
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning")
    if reasoning is None:
        reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str):
        reasoning = reasoning.strip()
    if not reasoning:
        reasoning = None
    return str(content), reasoning


def _build_request_messages(item: Dict[str, Any], system_prompt: str = "") -> List[Dict[str, str]]:
    raw_messages = item.get("messages")
    if isinstance(raw_messages, list) and raw_messages:
        return raw_messages
    prompt_text = str(item.get("prompt", ""))
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt_text})
    return messages

@dataclass
class APIConfig:
    """
    Configuration for the connection layer.
    Separates 'How to connect' from 'What to generate'.
    """
    api_key: str
    base_url: str
    model: str
    timeout: int = 6000
    max_retries: int = 500

# ------------------------- Async HTTP Client ------------------------

class AsyncClient:
    """
    Handles the raw HTTP transport.
    """
    def __init__(self, config: APIConfig):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        self.connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def post_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sends the payload exactly as constructed by the worker.
        Handles retries for network/server errors.
        """
        url = f"{self.config.base_url}/chat/completions"
        retry_delay = 1.0

        for attempt in range(self.config.max_retries + 1):
            try:
                async with self.session.post(url, json=payload) as response:
                    if response.status == 200:
                        return await response.json()
                    
                    # Retry on Rate Limit or Server Error
                    if response.status in [429, 500, 502, 503, 504]:
                        # Optional: Parse Retry-After header
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2 
                        continue
                    
                    # Hard fail on 400/401
                    text = await response.text()
                    raise ValueError(f"Client Error {response.status}: {text}")

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Network error: {str(e)}. Retrying...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2

        raise RuntimeError(f"Failed after {self.config.max_retries} retries.")

# ------------------------- Pipeline Engine ------------------------

class OnlineBatchInferenceEngine:
    """
    Orchestrates the pipeline with Unified Sampling Parameters.
    """

    def __init__(self, api_config: APIConfig, concurrency: int = 100):
        self.api_config = api_config
        self.concurrency = concurrency
        self.input_queue = asyncio.Queue(maxsize=concurrency * 2)
        self.output_queue = asyncio.Queue(maxsize=concurrency * 2)
        self.sem = asyncio.Semaphore(concurrency)

    # ------------------------- Worker Logic ------------------------

    async def _worker(self, client: AsyncClient, sampling_params: Dict[str, Any]):
        """
        Merges global sampling_params with per-request data.
        Strategy: Global Defaults < Per-Request Overrides
        """
        while True:
            item = await self.input_queue.get()
            if item is None:
                self.input_queue.task_done()
                break

            async with self.sem:
                start_t = time.perf_counter()
                try:
                    # 1. Construct Base Payload
                    system_prompt = str(sampling_params.get("__system_prompt", "") or "")
                    messages = _build_request_messages(item, system_prompt=system_prompt)
                    effective_sampling_params = {
                        key: value
                        for key, value in sampling_params.items()
                        if key != "__system_prompt"
                    }
                    
                    payload = {
                        "model": self.api_config.model,
                        "messages": messages,
                    }

                    # 2. Apply Global Sampling Params (Unified Management)
                    # This ensures consistency with offline engine behavior
                    payload.update(effective_sampling_params)

                    # 3. Apply Per-Request Overrides (Optional)
                    # If the input line has 'temperature', it overrides the global setting
                    # Filter input keys to avoid polluting payload with metadata like "id"
                    valid_overrides = {
                        k: v for k, v in item.items() 
                        if k in ["temperature", "max_tokens", "top_p", "stop", "frequency_penalty"]
                    }
                    payload.update(valid_overrides)

                    # 4. Execute
                    response = await client.post_request(payload)
                    
                    # 5. Result Formatting
                    content, reasoning = _extract_message_content_and_reasoning(response)
                    item["response"] = content
                    if reasoning:
                        # Preserve model reasoning traces when backend provides them.
                        item["thinking"] = reasoning
                    item["usage"] = response.get("usage", {})
                    item["_latency"] = round(time.perf_counter() - start_t, 3)
                    item["_status"] = "success"

                except Exception as e:
                    logger.error(f"Worker failed for ID {item.get('id', 'unknown')}: {e}")
                    item["_error"] = str(e)
                    item["_status"] = "failed"
                
                await self.output_queue.put(item)
                self.input_queue.task_done()

    # ------------------------- Helper Tasks ------------------------

    async def _producer(self, input_path: str, existing_ids: Set[str]):
        # Same as before: Read file -> Queue
        logger.info(f"Producer: Reading from {input_path}")
        f_in = sys.stdin if input_path == '-' else open(input_path, 'r', encoding='utf-8')
        try:
            for line in f_in:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    if existing_ids and data.get("id") in existing_ids: continue
                    await self.input_queue.put(data)
                except json.JSONDecodeError: pass
        finally:
            if input_path != '-': f_in.close()
        
        for _ in range(self.concurrency):
            await self.input_queue.put(None)

    async def _writer(self, output_path: str, total: int):
        # Same as before: Queue -> File
        pbar = tqdm(total=total, desc="Processing", unit="req")
        with open(output_path, 'a', encoding='utf-8') as f_out:
            while True:
                result = await self.output_queue.get()
                if result is None:
                    self.output_queue.task_done()
                    break
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()
                pbar.update(1)
                self.output_queue.task_done()
        pbar.close()

    # ------------------------- Main Run Interface ------------------------

    async def run(
        self, 
        input_file: str, 
        output_file: str, 
        sampling_params: Dict[str, Any]
    ):
        """
        Mirrors the offline engine's run signature.
        """
        # Resume Logic
        existing_ids = set()
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try: existing_ids.add(json.loads(line).get("id"))
                    except: pass
        
        # Count lines logic (omitted for brevity, same as before)
        total_lines = 0
        if input_file != '-':
            try: total_lines = int(os.popen(f'wc -l {input_file}').read().split()[0])
            except: pass
        remaining = max(0, total_lines - len(existing_ids))

        logger.info(f"Starting with Sampling Params: {sampling_params}")

        async with AsyncClient(self.api_config) as client:
            tasks = [
                asyncio.create_task(self._producer(input_file, existing_ids)),
                asyncio.create_task(self._writer(output_file, remaining if input_file != '-' else None))
            ]
            
            # Pass sampling_params explicitly to workers
            workers = [
                asyncio.create_task(self._worker(client, sampling_params)) 
                for _ in range(self.concurrency)
            ]
            
            await tasks[0] # Wait producer
            await self.input_queue.join()
            await self.output_queue.put(None) # Stop writer
            await tasks[1] # Wait writer
            
            for w in workers: w.cancel()

# ------------------------- CLI Entry Point ------------------------

def main():
    parser = argparse.ArgumentParser(description="Online Inference with Unified Sampling Params")
    
    # Connection Args
    parser.add_argument("--api-key", type=str, required=True, help="API Key")
    parser.add_argument("--base-url", type=str, required=True, help="API Base URL")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    
    # I/O Args
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--concurrency", type=int, default=50)

    # ------------------------- Sampling Params (Exposed) ------------------------
    # These mimic the keys usually found in 'sampling_params' dicts
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--stop", type=str, help="Stop sequence (comma separated if multiple, or raw string)")
    
    # Advanced: Allow passing a raw JSON string for obscure params (e.g. frequency_penalty)
    parser.add_argument("--extra-params", type=str, default="{}", help="JSON string for extra sampling params")

    args = parser.parse_args()

    # 1. Build Sampling Params Dictionary
    # This creates the 'unified' dictionary similar to the offline engine
    sampling_params = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p
    }
    
    # Handle Stop tokens
    if args.stop:
        # Simple heuristic: if comma exists, split, else single string
        if "," in args.stop:
            sampling_params["stop"] = args.stop.split(",")
        else:
            sampling_params["stop"] = args.stop

    # Merge extra JSON params
    if args.extra_params:
        try:
            extras = json.loads(args.extra_params)
            sampling_params.update(extras)
        except json.JSONDecodeError:
            logger.error("Failed to parse --extra-params JSON")
            sys.exit(1)

    # 2. Init Config
    config = APIConfig(
        api_key=args.api_key,
        base_url=args.base_url.rstrip('/'),
        model=args.model
    )

    engine = OnlineBatchInferenceEngine(config, concurrency=args.concurrency)

    # 3. Run
    try:
        asyncio.run(engine.run(args.input, args.output, sampling_params))
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()