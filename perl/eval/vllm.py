import argparse
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import aiohttp
from typing import List, Tuple, Iterable
from pathlib import Path


def extract_vllm_args(unknown: List[str]) -> Tuple[List[str], List[str]]:
    vllm_args: List[str] = []
    leftover: List[str] = []
    idx = 0
    while idx < len(unknown):
        token = unknown[idx]
        if token.startswith("--vllm-"):
            stripped = "--" + token[len("--vllm-") :]
            if "=" in token:
                _, value = token.split("=", 1)
                vllm_args.extend([stripped, value])
            elif idx + 1 < len(unknown) and not unknown[idx + 1].startswith("-"):
                vllm_args.extend([stripped, unknown[idx + 1]])
                idx += 1
            else:
                vllm_args.append(stripped)
        else:
            leftover.append(token)
        idx += 1
    return vllm_args, leftover


def build_vllm_command(
    model_path: Path, port: int, args: argparse.Namespace, vllm_args: List[str]
) -> List[str]:
    dp_size = max(1, args.dp_size)
    max_concurrent_per_dp = max(1, args.max_num_request // dp_size)

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_path),
        "--served-model-name",
        args.served_model_name,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(args.tp_size),
        "--max-num-seqs",
        str(max_concurrent_per_dp),
    ]

    # Solution: append --gpu-memory-utilization to vLLM command, default 0.95, configurable via CLI.
    if args.gpu_memory_utilization is not None:
        cmd.extend(["--gpu-memory-utilization", str(args.gpu_memory_utilization)])
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    cmd.extend(vllm_args)
    return cmd


def pipe_to_logger(
    stream: Iterable[str], logger: logging.Logger, level: int, prefix: str
) -> None:
    for line in stream:
        logger.log(level, "%s%s", prefix, line.rstrip("\n"))


def start_vllm_processes(
    model_path: Path,
    args: argparse.Namespace,
    vllm_args: List[str],
    logger: logging.Logger,
) -> Tuple[List[subprocess.Popen], List[int]]:
    ports: List[int] = []
    processes: List[subprocess.Popen] = []
    env = os.environ.copy()
    dp_size = max(1, args.dp_size)

    for rank in range(dp_size):
        # Calculate GPU ID range for current process
        start_gpu_id = rank * args.tp_size
        end_gpu_id = start_gpu_id + args.tp_size
        gpu_ids = list(range(start_gpu_id, end_gpu_id))

        # Check for out-of-bounds (based on args.num_gpus or simple logic, assuming user config is correct)

        env_local = env.copy()
        env_local["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

        port = args.serve_port + rank
        cmd = build_vllm_command(model_path, port, args, vllm_args)
        logger.info(
            "Starting vLLM backend [%d/%d], port %d, GPUs=%s, command: %s",
            rank + 1,
            dp_size,
            port,
            gpu_ids,
            " ".join(cmd),
        )
        proc = subprocess.Popen(
            cmd,
            env=env_local,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )
        processes.append(proc)
        ports.append(port)
        if proc.stdout:
            threading.Thread(
                target=pipe_to_logger,
                args=(proc.stdout, logger, logging.INFO, f"[vllm:{port}] "),
                daemon=True,
            ).start()
        if proc.stderr:
            threading.Thread(
                target=pipe_to_logger,
                args=(proc.stderr, logger, logging.ERROR, f"[vllm:{port}] "),
                daemon=True,
            ).start()
    return processes, ports


def stop_vllm_processes(
    processes: List[subprocess.Popen], logger: logging.Logger
) -> None:
    for proc in processes:
        if proc.poll() is None:
            try:
                logger.info("Attempting to terminate vLLM process (pid=%d).", proc.pid)
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Exception during process termination (pid=%d): %s", proc.pid, exc
                )
    for proc in processes:
        if proc.poll() is None:
            try:
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass


def wait_for_vllm_ready(
    port: int, process: subprocess.Popen, timeout: float, logger: logging.Logger
) -> bool:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if process.poll() is not None:
            logger.error("vLLM process (pid=%d) exited prematurely.", process.pid)
            return False
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    logger.info("vLLM at port %d is ready.", port)
                    return True
        except Exception:
            time.sleep(2)
    logger.error("Timeout waiting for vLLM at port %d.", port)
    return False


async def generate_with_vllm_async(
    session: aiohttp.ClientSession, prompt: str, port: int, args: argparse.Namespace
) -> str:
    """Async version of vLLM generation function for concurrent requests."""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": args.served_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_new_tokens,
        "n": 1,
    }
    if args.seed is not None:
        payload["seed"] = args.seed
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    try:
        async with session.post(
            url, json=payload, headers=headers, timeout=timeout
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"vLLM returned HTTP error: {response.status}")
            content = await response.json()
    except aiohttp.ClientError as exc:
        raise RuntimeError(f"vLLM connection failed: {exc}") from exc

    try:
        return content["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to parse vLLM response: {content}") from exc
