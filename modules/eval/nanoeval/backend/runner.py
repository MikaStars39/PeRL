from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _mock_infer_response(record: Dict[str, Any]) -> str:
    if "mock_response" in record:
        return str(record["mock_response"])
    prompt = str(record.get("prompt", ""))
    return f"[MOCK_RESPONSE] {prompt[:120]}"


def _run_mock_backend(input_file: Path, output_file: Path) -> Dict[str, Any]:
    input_rows = _read_jsonl(input_file)
    output_rows: List[Dict[str, Any]] = []

    for row in input_rows:
        result_row = dict(row)
        result_row["response"] = _mock_infer_response(row)
        output_rows.append(result_row)

    count = _write_jsonl(output_file, output_rows)
    return {
        "backend": "mock",
        "input_count": len(input_rows),
        "output_count": count,
        "output_path": str(output_file),
    }


def run_inference(
    backend: str,
    input_file: Path,
    output_file: Path,
    sampling_params: Optional[Dict[str, Any]] = None,
    model_path: Optional[str] = None,
    tp_size: int = 1,
    dp_size: int = 1,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    concurrency: int = 32,
    ray_num_actors: Optional[int] = None,
    ray_worker_concurrency: Optional[int] = None,
    online_request_timeout_s: float = 120.0,
    online_stall_log_interval_s: float = 15.0,
) -> Dict[str, Any]:
    backend_name = backend.strip().lower().replace("-", "_")
    params = sampling_params or {}

    if backend_name == "mock":
        return _run_mock_backend(input_file=input_file, output_file=output_file)

    if backend_name == "offline":
        if not model_path:
            raise ValueError("model_path is required when backend=offline")
        from .offline import BatchInferenceEngine

        # SGLang offline sampling uses `max_new_tokens` instead of `max_tokens`.
        offline_params = dict(params)
        if "max_tokens" in offline_params and "max_new_tokens" not in offline_params:
            offline_params["max_new_tokens"] = offline_params.pop("max_tokens")

        async def _run_offline() -> None:
            async with BatchInferenceEngine(
                model_path=model_path,
                tp_size=max(1, int(tp_size)),
                dp_size=max(1, int(dp_size)),
            ) as engine:
                await engine.run(
                    input_file=str(input_file),
                    output_file=str(output_file),
                    sampling_params=offline_params,
                    resume=False,
                )

        asyncio.run(_run_offline())
        return {
            "backend": "offline",
            "input_path": str(input_file),
            "output_path": str(output_file),
        }

    if backend_name == "online":
        if not api_key or not base_url or not model:
            raise ValueError("api_key, base_url and model are required when backend=online")
        from .online import APIConfig, OnlineBatchInferenceEngine

        async def _run_online() -> None:
            config = APIConfig(
                api_key=api_key,
                base_url=base_url.rstrip("/"),
                model=model,
            )
            engine = OnlineBatchInferenceEngine(config, concurrency=concurrency)
            await engine.run(
                input_file=str(input_file),
                output_file=str(output_file),
                sampling_params=params,
            )

        asyncio.run(_run_online())
        return {
            "backend": "online",
            "input_path": str(input_file),
            "output_path": str(output_file),
        }

    if backend_name == "online_ray":
        if not api_key or not base_url or not model:
            raise ValueError("api_key, base_url and model are required when backend=online_ray")
        from .online_ray import APIConfig, OnlineRayBatchInferenceEngine
        actor_count = (
            max(1, int(ray_num_actors))
            if ray_num_actors is not None
            else max(1, int(concurrency) // 8)
        )
        worker_concurrency = (
            max(1, int(ray_worker_concurrency))
            if ray_worker_concurrency is not None
            else max(1, int(concurrency) // actor_count)
        )

        async def _run_online_ray() -> None:
            config = APIConfig(
                api_key=api_key,
                base_url=base_url.rstrip("/"),
                model=model,
            )
            engine = OnlineRayBatchInferenceEngine(
                config,
                num_actors=actor_count,
                worker_concurrency=worker_concurrency,
                request_timeout_s=online_request_timeout_s,
                stall_log_interval_s=online_stall_log_interval_s,
            )
            await engine.run(
                input_file=str(input_file),
                output_file=str(output_file),
                sampling_params=params,
            )

        asyncio.run(_run_online_ray())
        return {
            "backend": "online_ray",
            "input_path": str(input_file),
            "output_path": str(output_file),
            "ray_num_actors": actor_count,
            "ray_worker_concurrency": worker_concurrency,
            "online_request_timeout_s": online_request_timeout_s,
            "online_stall_log_interval_s": online_stall_log_interval_s,
        }

    raise ValueError(f"Unsupported backend: {backend}")
