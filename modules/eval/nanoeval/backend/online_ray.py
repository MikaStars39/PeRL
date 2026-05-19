from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Set

logger = logging.getLogger("OnlineRayInference")

try:
    import ray
except ImportError:  # pragma: no cover - covered by runtime guard.
    ray = None


@dataclass
class APIConfig:
    api_key: str
    base_url: str
    model: str
    timeout: int = 6000
    max_retries: int = 500


def _get_async_client_cls():
    try:
        from .online import AsyncClient
    except Exception as exc:
        raise RuntimeError(
            "online_ray requires online backend dependencies (e.g. aiohttp)."
        ) from exc
    return AsyncClient


def _count_lines_fast(file_path: str) -> int:
    if not os.path.exists(file_path):
        return 0
    total = 0
    with open(file_path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if line.strip():
                total += 1
    return total


def _load_existing_ids(path: str, success_only: bool = True) -> Set[str]:
    existing_ids: Set[str] = set()
    if not os.path.exists(path):
        return existing_ids
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if success_only and payload.get("_status") not in (None, "success"):
                continue
            row_id = payload.get("id")
            if isinstance(row_id, str):
                existing_ids.add(row_id)
    return existing_ids


def _read_pending_rows(input_file: str, existing_ids: Set[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(input_file, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if existing_ids and row.get("id") in existing_ids:
                continue
            rows.append(row)
    return rows


def _chunk_rows(rows: List[Dict[str, Any]], chunk_size: int) -> List[List[Dict[str, Any]]]:
    size = max(1, int(chunk_size))
    return [rows[idx : idx + size] for idx in range(0, len(rows), size)]


def _extract_message_content_and_reasoning(response: Dict[str, Any]) -> tuple[str, str | None]:
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


if ray is not None:

    @ray.remote
    class OnlineRayWorker:
        def __init__(
            self,
            api_config_dict: Dict[str, Any],
            worker_concurrency: int,
            sampling_params: Dict[str, Any],
            request_timeout_s: float = 120.0,
        ):
            self.api_config = APIConfig(**api_config_dict)
            self.worker_concurrency = max(1, int(worker_concurrency))
            self.sampling_params = dict(sampling_params)
            self.request_timeout_s = max(1.0, float(request_timeout_s))

        async def run_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            sem = asyncio.Semaphore(self.worker_concurrency)
            results: List[Dict[str, Any]] = [dict() for _ in range(len(rows))]
            async_client_cls = _get_async_client_cls()
            async with async_client_cls(self.api_config) as client:
                tasks = [
                    asyncio.create_task(self._infer_one(client, sem, idx, row, results))
                    for idx, row in enumerate(rows)
                ]
                if tasks:
                    await asyncio.gather(*tasks)
            return results

        async def _infer_one(
            self,
            client: Any,
            sem: asyncio.Semaphore,
            idx: int,
            row: Dict[str, Any],
            results: List[Dict[str, Any]],
        ) -> None:
            async with sem:
                item = dict(row)
                started_at = time.perf_counter()
                try:
                    # Local testing shortcut to avoid remote API dependency.
                    if "mock_response" in item:
                        item["response"] = str(item["mock_response"])
                        item["usage"] = {}
                        item["_latency"] = round(time.perf_counter() - started_at, 3)
                        item["_status"] = "success"
                        results[idx] = item
                        return

                    system_prompt = str(self.sampling_params.get("__system_prompt", "") or "")
                    messages = _build_request_messages(item, system_prompt=system_prompt)
                    payload = {
                        "model": self.api_config.model,
                        "messages": messages,
                    }
                    payload.update(
                        {
                            key: value
                            for key, value in self.sampling_params.items()
                            if key != "__system_prompt"
                        }
                    )
                    payload.update(
                        {
                            key: value
                            for key, value in item.items()
                            if key
                            in {
                                "temperature",
                                "max_tokens",
                                "top_p",
                                "stop",
                                "frequency_penalty",
                                "presence_penalty",
                                "repetition_penalty",
                            }
                        }
                    )
                    response = await asyncio.wait_for(
                        client.post_request(payload),
                        timeout=self.request_timeout_s,
                    )
                    content, reasoning = _extract_message_content_and_reasoning(response)
                    item["response"] = content
                    if reasoning:
                        # Preserve model reasoning traces when backend provides them.
                        item["thinking"] = reasoning
                    item["usage"] = response.get("usage", {})
                    item["_latency"] = round(time.perf_counter() - started_at, 3)
                    item["_status"] = "success"
                except asyncio.TimeoutError:
                    item["_status"] = "failed"
                    item["_error"] = f"request timeout after {self.request_timeout_s:.1f}s"
                    item["_latency"] = round(time.perf_counter() - started_at, 3)
                except Exception as exc:
                    item["_status"] = "failed"
                    item["_error"] = str(exc)
                    item["_latency"] = round(time.perf_counter() - started_at, 3)
                results[idx] = item


class OnlineRayBatchInferenceEngine:
    """
    Multi-process online inference engine powered by Ray actors.
    """

    def __init__(
        self,
        api_config: APIConfig,
        num_actors: int = 2,
        worker_concurrency: int = 16,
        request_timeout_s: float = 120.0,
        stall_log_interval_s: float = 15.0,
    ):
        self.api_config = api_config
        self.num_actors = max(1, int(num_actors))
        self.worker_concurrency = max(1, int(worker_concurrency))
        self.request_timeout_s = max(1.0, float(request_timeout_s))
        self.stall_log_interval_s = max(1.0, float(stall_log_interval_s))

    async def run(self, input_file: str, output_file: str, sampling_params: Dict[str, Any]) -> None:
        if ray is None:
            raise RuntimeError("ray is required for online_ray backend. Please install ray first.")

        existing_ids = _load_existing_ids(output_file, success_only=True)
        total = _count_lines_fast(input_file)
        pending_rows = _read_pending_rows(input_file, existing_ids=existing_ids)
        if not pending_rows:
            logger.info("Nothing to process for online_ray backend.")
            return

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, include_dashboard=False, logging_level="ERROR")

        started_at = time.perf_counter()
        actor_count = min(self.num_actors, len(pending_rows))
        chunk_size = max(8, min(32, self.worker_concurrency))
        row_chunks = _chunk_rows(pending_rows, chunk_size=chunk_size)
        total_chunks = len(row_chunks)
        workers = [
            OnlineRayWorker.options(num_cpus=1).remote(
                api_config_dict=asdict(self.api_config),
                worker_concurrency=self.worker_concurrency,
                sampling_params=dict(sampling_params),
                request_timeout_s=self.request_timeout_s,
            )
            for _ in range(actor_count)
        ]
        logger.info(
            "online_ray start: pending=%s actors=%s worker_concurrency=%s chunk_size=%s total_chunks=%s",
            len(pending_rows),
            actor_count,
            self.worker_concurrency,
            chunk_size,
            total_chunks,
        )
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        written = 0
        completed_chunks = 0
        inflight: Dict[Any, int] = {}
        next_chunk_idx = 0
        for worker_idx in range(actor_count):
            if next_chunk_idx >= total_chunks:
                break
            ref = workers[worker_idx].run_rows.remote(row_chunks[next_chunk_idx])
            inflight[ref] = worker_idx
            next_chunk_idx += 1

        with open(output_file, "a", encoding="utf-8") as file_obj:
            while inflight:
                done_refs, _ = ray.wait(
                    list(inflight.keys()),
                    num_returns=1,
                    timeout=self.stall_log_interval_s,
                )
                if not done_refs:
                    elapsed = max(1e-9, time.perf_counter() - started_at)
                    logger.warning(
                        "online_ray heartbeat: no completed chunk in last %.1fs, inflight=%s, dispatched=%s/%s, written=%s, speed=%.2f req/s",
                        self.stall_log_interval_s,
                        len(inflight),
                        next_chunk_idx,
                        total_chunks,
                        written,
                        written / elapsed,
                    )
                    continue
                done_ref = done_refs[0]
                worker_idx = inflight.pop(done_ref)
                batch = ray.get(done_ref)
                for row in batch:
                    file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
                file_obj.flush()
                completed_chunks += 1

                if next_chunk_idx < total_chunks:
                    new_ref = workers[worker_idx].run_rows.remote(row_chunks[next_chunk_idx])
                    inflight[new_ref] = worker_idx
                    next_chunk_idx += 1

                if completed_chunks % 10 == 0 or completed_chunks == total_chunks:
                    elapsed = max(1e-9, time.perf_counter() - started_at)
                    logger.info(
                        "online_ray progress: chunks=%s/%s written=%s/%s speed=%.2f req/s",
                        completed_chunks,
                        total_chunks,
                        written,
                        len(pending_rows),
                        written / elapsed,
                    )

        elapsed = max(1e-9, time.perf_counter() - started_at)
        logger.info(
            "online_ray completed: written=%s pending=%s total=%s throughput=%.2f req/s",
            written,
            len(pending_rows),
            total,
            written / elapsed,
        )
