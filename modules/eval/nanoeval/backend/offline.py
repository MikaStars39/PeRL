import json
import asyncio
import time
import os
import subprocess
import logging
from typing import Optional, Dict, Any, Set, List, Callable, Tuple
from tqdm.asyncio import tqdm
from .base import BaseSGLangEngine

logger = logging.getLogger("BatchInference")

class BatchInferenceEngine(BaseSGLangEngine):
    """
    A robust pipeline for high-throughput offline inference.
    Adds Producer/Worker/Writer queues on top of BaseEngine.
    """

    def __init__(
        self,
        max_inflight: int = 512,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.max_inflight = max_inflight
        self.input_queue: Optional[asyncio.Queue] = None
        self.output_queue: Optional[asyncio.Queue] = None

    # ------------------------- Lifecycle Override ------------------------

    async def __aenter__(self):
        """Call super init and setup queues."""
        await super().__aenter__()
        self.input_queue = asyncio.Queue(maxsize=self.max_inflight * 2)
        self.output_queue = asyncio.Queue(maxsize=self.max_inflight * 2)
        return self

    # ------------------------- Helper ------------------------

    @staticmethod
    def _count_lines_fast(file_path: str) -> int:
        """Counts lines using 'wc -l' with fallback."""
        if not os.path.exists(file_path): return 0
        try:
            res = subprocess.run(['wc', '-l', file_path], capture_output=True, text=True)
            return int(res.stdout.split()[0])
        except Exception:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f if _.strip())

    def _extract_stats(self, output: Dict[str, Any]) -> Dict[str, int]:
        """Extracts token counts from SGLang output."""
        meta = output.get("meta_info") or output.get("usage") or {}
        pt = meta.get("prompt_tokens") or meta.get("input_tokens") or 0
        ct = meta.get("completion_tokens") or meta.get("output_tokens") or 0
        return {"prompt": int(pt), "completion": int(ct), "total": int(pt + ct)}

    # ------------------------- Async Tasks ------------------------

    async def _producer(self, input_file: str, existing_ids: Set[str]):
        """Reads input file and fills input_queue."""
        logger.info("Producer: Started reading file...")
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    if existing_ids and data.get("id") in existing_ids: continue
                    await self.input_queue.put(data)
                except json.JSONDecodeError: pass
        
        # Poison pills
        for _ in range(self.max_inflight):
            await self.input_queue.put(None)

    async def _worker(self, sampling_params: dict):
        """Consumes inputs, calls base._generate_safe, pushes to output."""
        while True:
            item = await self.input_queue.get()
            if item is None:
                self.input_queue.task_done()
                break
            
            try:
                start_t = time.perf_counter()
                # Call the robust generation method from Base Class
                output = await self._generate_safe(item["prompt"], sampling_params)
                
                item["response"] = output["text"]
                item["_stats"] = self._extract_stats(output)
                item["_latency"] = time.perf_counter() - start_t
                
                await self.output_queue.put(item)
            except Exception as e:
                logger.error(f"Worker Error: {e}")
            finally:
                self.input_queue.task_done()

    async def _writer(self, output_file: str, total_items: int, resume: bool):
        """Writes results to disk with progress bar."""
        mode = "a" if resume else "w"
        pbar = tqdm(total=total_items, desc="Inference")
        acc_tokens, start_global = 0, time.perf_counter()
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, mode, encoding='utf-8') as f_out:
            while True:
                result = await self.output_queue.get()
                if result is None:
                    self.output_queue.task_done()
                    break
                
                stats = result.pop("_stats", {})
                result.pop("_latency", None)
                acc_tokens += stats.get("total", 0)
                
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                
                # UI Update
                pbar.update(1)
                elapsed = max(1e-9, time.perf_counter() - start_global)
                pbar.set_postfix({"tok/s": f"{acc_tokens/elapsed:.1f}"})
                self.output_queue.task_done()
        pbar.close()

    # ------------------------- Main Entry Point ------------------------

    async def run(self, input_file: str, output_file: str, sampling_params: dict, resume: bool = False):
        """Orchestrates the pipeline."""
        # 1. Resume Logic
        existing_ids = set()
        if resume and os.path.exists(output_file):
            logger.info("Scanning output for resume...")
            with open(output_file, "r") as f:
                for line in f:
                    try: existing_ids.add(json.loads(line).get("id"))
                    except: pass
        
        # 2. Workload Check
        total_lines = self._count_lines_fast(input_file)
        remaining = max(0, total_lines - len(existing_ids))
        if remaining == 0 and total_lines > 0:
            logger.info("Nothing to do.")
            return

        # 3. Launch Tasks
        tasks = [
            asyncio.create_task(self._producer(input_file, existing_ids)),
            asyncio.create_task(self._writer(output_file, remaining, resume))
        ]
        tasks += [asyncio.create_task(self._worker(sampling_params)) for _ in range(self.max_inflight)]

        # 4. Wait
        await tasks[0] # Wait for producer
        await self.input_queue.join()
        await self.output_queue.put(None) # Stop writer
        await tasks[1] # Wait for writer
        
        logger.info(f"Done. Saved to {output_file}")

    # ------------------------- Multi-Turn Support ------------------------

    async def _multi_turn_worker(
        self, 
        sampling_params: dict, 
        turn_callback: Callable[[List[str], str], Tuple[bool, Optional[str]]],
        max_turns: int
    ):
        """Handles multi-turn conversation for a single item."""
        while True:
            item = await self.input_queue.get()
            if item is None:
                self.input_queue.task_done()
                break
            
            try:
                start_t = time.perf_counter()
                conversation: List[str] = []
                prompt = item["prompt"]
                total_stats = {"prompt": 0, "completion": 0, "total": 0}
                messages: dict = {}
                
                for _ in range(max_turns):
                    output = await self._generate_safe(prompt, sampling_params)
                    conversation.append(output)
                    
                    # Accumulate stats
                    stats = self._extract_stats(output)
                    for k in total_stats: total_stats[k] += stats[k]
                    
                    # Check if should continue
                    should_continue, next_prompt, messages = turn_callback(conversation, messages)
                    if not should_continue or next_prompt is None:
                        break
                    prompt = next_prompt
                
                item["responses"] = conversation
                item["_stats"] = total_stats
                item["_latency"] = time.perf_counter() - start_t
                
                await self.output_queue.put(item)
            except Exception as e:
                logger.error(f"MultiTurn Worker Error: {e}")
            finally:
                self.input_queue.task_done()

    async def run_multi_turn(
        self, 
        input_file: str, 
        output_file: str, 
        sampling_params: dict,
        turn_callback: Callable[[List[str], str], Tuple[bool, Optional[str]]],
        max_turns: int = 10,
        resume: bool = False
    ):
        """
        Multi-turn inference pipeline.
        
        Args:
            turn_callback: (conversation_history, latest_response) -> (should_continue, next_prompt)
                           Return (False, None) to stop, (True, prompt) to continue.
            max_turns: Maximum turns per conversation to prevent infinite loops.
        """
        # 1. Resume Logic
        existing_ids = set()
        if resume and os.path.exists(output_file):
            logger.info("Scanning output for resume...")
            with open(output_file, "r") as f:
                for line in f:
                    try: existing_ids.add(json.loads(line).get("id"))
                    except: pass
        
        # 2. Workload Check
        total_lines = self._count_lines_fast(input_file)
        remaining = max(0, total_lines - len(existing_ids))
        if remaining == 0 and total_lines > 0:
            logger.info("Nothing to do.")
            return

        # 3. Launch Tasks
        tasks = [
            asyncio.create_task(self._producer(input_file, existing_ids)),
            asyncio.create_task(self._writer(output_file, remaining, resume))
        ]
        tasks += [
            asyncio.create_task(self._multi_turn_worker(sampling_params, turn_callback, max_turns)) 
            for _ in range(self.max_inflight)
        ]

        # 4. Wait
        await tasks[0]  # Wait for producer
        await self.input_queue.join()
        await self.output_queue.put(None)  # Stop writer
        await tasks[1]  # Wait for writer
        
        logger.info(f"Done. Saved to {output_file}")