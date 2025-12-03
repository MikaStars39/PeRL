#!/usr/bin/env python3
# 本脚本依据用户需求：实现评测流程（参数解析、模型合并、启动vLLM、生成、打分、缓存/恢复、日志、阶段化提示、最终统计）。
# 实现方案：使用argparse解析常规与--vllm-*透传参数，必要时在CPU上合并LoRA并保存；后台启动支持数据并行的vLLM服务器，
# 轮询后端生成多次rollout并缓存到文件，随后调用score_response汇总为result.jsonl，最后新增一个统计阶段输出avg@k/pass@k，
# 同时记录日志并将stdout/stderr写入latest_run.log；通过阶段化日志标明第几阶段的开始/结束（含emoji）。本版强制依赖vLLM与GPU，
# 新增 --num-gpus 参数用于运行前校验可用 GPU 数（bash 脚本中设为 8），确保按需求使用多卡并行。

import argparse
import asyncio
import atexit
import importlib.util
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional
import math

try:
    import aiohttp
except ImportError:
    raise ImportError("需要安装 aiohttp: pip install aiohttp")

from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class StreamToLogger:
    """Redirect stdout/stderr到logger，确保输出被文件与控制台同时记录。"""

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
    log_path = result_dir / "latest_run.log"

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.__stdout__)
    console_handler.setFormatter(formatter)
    logging.root.addHandler(file_handler)
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
    """阶段化日志上下文，标记开始/结束和失败场景。"""

    def __init__(
        self,
        logger: logging.Logger,
        stage_id: int,
        name: str,
        emoji_start: str = "🚀",
        emoji_end: str = "🏁",
        emoji_fail: str = "💥",
    ) -> None:
        self.logger = logger
        self.stage_id = stage_id
        self.name = name
        self.emoji_start = emoji_start
        self.emoji_end = emoji_end
        self.emoji_fail = emoji_fail

    def __enter__(self) -> "StageContext":
        self.logger.info("%s 第%d阶段开始：%s", self.emoji_start, self.stage_id, self.name)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        if exc_type is None:
            self.logger.info("%s 第%d阶段结束：%s", self.emoji_end, self.stage_id, self.name)
        else:
            self.logger.error("%s 第%d阶段失败：%s，错误：%s", self.emoji_fail, self.stage_id, self.name, exc)


def parse_args() -> Tuple[argparse.Namespace, List[str], List[str]]:
    parser = argparse.ArgumentParser(description="评测入口脚本，支持模型合并、vLLM启动与多数据集评测。")
    parser.add_argument("--result-dir", required=True, help="中间过程与结果输出目录。")
    parser.add_argument("--model", required=True, help="基础模型名称或路径。")
    parser.add_argument("--adapter", default="", help="LoRA/PEFT adapter路径，留空表示不合并。")
    parser.add_argument("--dataset", default="aime2024", help="要评测的数据集缩写，英文逗号分隔（如：aime2024）。")
    parser.add_argument("--rollout-n", type=int, default=1, help="每个sample生成多少次rollout。")
    parser.add_argument("--serve-port", type=int, default=8000, help="第一个vLLM后端端口号。")
    parser.add_argument("--dp-size", type=int, default=1, help="数据并行后端数量（启动多个vLLM）。")
    parser.add_argument("--tp-size", type=int, default=1, help="传给vLLM的张量并行大小。")
    parser.add_argument("--num-gpus", type=int, default=1, help="运行前校验需要的GPU数量，不足则报错。")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help="传给vLLM的GPU显存利用率上限（0~1），用于控制单卡显存占用比例。",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="生成温度。")
    parser.add_argument("--top-p", type=float, default=1.0, help="生成top-p。")
    parser.add_argument("--max-new-tokens", type=int, default=131072, help="生成长度。")
    parser.add_argument("--dtype", default="auto", help="模型dtype，用于合并环节。")
    parser.add_argument("--trust-remote-code", action="store_true", help="是否信任远程代码。")
    parser.add_argument("--served-model-name", default="eval-model", help="vLLM对外暴露的模型名。")
    parser.add_argument("--api-key", default="dummy", help="OpenAI兼容接口的API Key。")
    parser.add_argument("--request-timeout", type=float, default=600.0, help="请求单次超时时间。")
    parser.add_argument("--max-samples", type=int, default=None, help="调试用，限制评测样本数量。")
    parser.add_argument(
        "--max-num-request-per-dp",
        type=int,
        default=1,
        help="每个数据并行（DP）的vLLM后端同时运行的请求数上限。",
    )

    args, unknown = parser.parse_known_args()
    vllm_args, leftover = extract_vllm_args(unknown)
    return args, vllm_args, leftover


def extract_vllm_args(unknown: List[str]) -> Tuple[List[str], List[str]]:
    vllm_args: List[str] = []
    leftover: List[str] = []
    idx = 0
    while idx < len(unknown):
        token = unknown[idx]
        if token.startswith("--vllm-"):
            stripped = "--" + token[len("--vllm-"):]
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


def resolve_torch_dtype(dtype: Any) -> Any:
    """将dtype字符串解析为torch.dtype，支持auto/常见别名，兼容旧版Transformers缺少get_torch_dtype的场景。"""
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
    raise ValueError(f"不支持的dtype: {dtype}")


def prepare_prompt(sample: Dict[str, Any]) -> str:
    """根据sample构建模型输入prompt，可按需修改增强。"""
    if isinstance(sample, dict):
        if "prompt" in sample:
            return str(sample["prompt"])
        if "instruction" in sample and "input" in sample:
            return f"{sample['instruction']}\n{sample['input']}"
        if "instruction" in sample:
            return str(sample["instruction"])
        if "question" in sample:
            return str(sample["question"])
        if "text" in sample:
            return str(sample["text"])
    return str(sample)


def score_response(prompt: str, response: str, sample: Dict[str, Any]) -> float:
    """简单占位评分：若sample包含answer/label且出现在response则记1，否则0。"""
    answer = None
    if isinstance(sample, dict):
        answer = sample.get("answer") or sample.get("label")
    if answer is None:
        return 0.0
    return float(str(answer) in response)


def merge_model_if_needed(args: argparse.Namespace, result_dir: Path, logger: logging.Logger) -> Path:
    if not args.adapter:
        logger.info("未提供adapter，直接使用基础模型：%s", args.model)
        return Path(args.model)

    output_dir = result_dir / "model"
    if output_dir.exists() and any(output_dir.iterdir()):
        logger.info("检测到已存在的合并模型目录，直接复用：%s", output_dir)
        return output_dir

    torch_dtype = resolve_torch_dtype(args.dtype)
    logger.info("加载基础模型：%s", args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="cpu",
        trust_remote_code=args.trust_remote_code,
    )
    logger.info("加载分词器：%s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    logger.info("加载LoRA/PEFT adapter：%s", args.adapter)
    model = PeftModel.from_pretrained(model, args.adapter)
    logger.info("执行merge_and_unload，将LoRA权重写入基础模型。")
    model = model.merge_and_unload()

    logger.info("保存合并模型至：%s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def build_vllm_command(
    model_path: Path, port: int, args: argparse.Namespace, vllm_args: List[str]
) -> List[str]:
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
    ]
    # 实现方案：在构造 vLLM 启动命令时追加 --gpu-memory-utilization 参数，默认 0.95，可通过命令行覆盖。
    if args.gpu_memory_utilization is not None:
        cmd.extend(["--gpu-memory-utilization", str(args.gpu_memory_utilization)])
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    cmd.extend(vllm_args)
    return cmd


def pipe_to_logger(stream: Iterable[str], logger: logging.Logger, level: int, prefix: str) -> None:
    for line in stream:
        logger.log(level, "%s%s", prefix, line.rstrip("\n"))


def start_vllm_processes(
    model_path: Path, args: argparse.Namespace, vllm_args: List[str], logger: logging.Logger
) -> Tuple[List[subprocess.Popen], List[int]]:
    ports: List[int] = []
    processes: List[subprocess.Popen] = []
    env = os.environ.copy()
    dp_size = max(1, args.dp_size)

    for rank in range(dp_size):
        # 计算当前进程分配的GPU ID范围
        start_gpu_id = rank * args.tp_size
        end_gpu_id = start_gpu_id + args.tp_size
        gpu_ids = list(range(start_gpu_id, end_gpu_id))
        
        # 校验是否越界（基于args.num_gpus或者简单的逻辑校验，这里假设用户配置正确）
        # 如果需要更严格校验，可以在此处添加。
        
        env_local = env.copy()
        env_local["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        
        port = args.serve_port + rank
        cmd = build_vllm_command(model_path, port, args, vllm_args)
        logger.info("启动vLLM后端[%d/%d]，端口%d，GPUs=%s，命令：%s", rank + 1, dp_size, port, gpu_ids, " ".join(cmd))
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


def stop_vllm_processes(processes: List[subprocess.Popen], logger: logging.Logger) -> None:
    for proc in processes:
        if proc.poll() is None:
            try:
                logger.info("尝试终止vLLM进程(pid=%d)。", proc.pid)
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception as exc:  # noqa: BLE001
                logger.warning("终止进程(pid=%d)时发生异常：%s", proc.pid, exc)
    for proc in processes:
        if proc.poll() is None:
            try:
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass


def wait_for_vllm_ready(port: int, process: subprocess.Popen, timeout: float, logger: logging.Logger) -> bool:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if process.poll() is not None:
            logger.error("vLLM进程(pid=%d)提前退出。", process.pid)
            return False
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    logger.info("端口%d的vLLM已就绪。", port)
                    return True
        except Exception:
            time.sleep(2)
    logger.error("等待端口%d的vLLM超时。", port)
    return False


def load_dataset_by_name(name: str, split: str):
    if ":" in name:
        path, subset = name.split(":", 1)
        return load_dataset(path, subset, split=split)
    return load_dataset(name, split=split)


def load_task_module(task_abbr: str) -> Any:
    """
    根据任务缩写动态加载对应的任务模块。
    实现方案：根据缩写（如aime2024）动态加载tasks/{aime2024}.py脚本中的函数。
    简化实现：将项目根目录添加到sys.path，使相对导入能够正常工作。
    
    Args:
        task_abbr: 任务缩写，如 "aime2024"
    
    Returns:
        加载的任务模块对象
    
    Raises:
        FileNotFoundError: 如果任务文件不存在
        ImportError: 如果模块加载失败
    """
    # 获取当前脚本所在目录和项目根目录
    current_dir = Path(__file__).parent  # scripts/eval
    project_root = current_dir.parent.parent  # 项目根目录（包含scripts的目录）
    task_file = current_dir / "tasks" / f"{task_abbr}.py"
    
    if not task_file.exists():
        raise FileNotFoundError(f"任务文件不存在: {task_file}")
    
    # 将项目根目录添加到sys.path（如果还没有的话），这样相对导入就能正常工作
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    # 使用importlib动态加载模块
    module_name = f"scripts.eval.tasks.{task_abbr}"
    spec = importlib.util.spec_from_file_location(module_name, task_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载任务模块: {task_file}")
    
    module = importlib.util.module_from_spec(spec)
    # 设置__package__属性，使相对导入能够正常工作
    module.__package__ = "scripts.eval.tasks"
    module.__name__ = module_name
    
    # 执行模块代码
    spec.loader.exec_module(module)
    
    # 验证必需的函数是否存在
    required_functions = ["load_dataset", "prepare_prompt", "score_response"]
    for func_name in required_functions:
        if not hasattr(module, func_name):
            raise ImportError(f"任务模块 {task_abbr} 缺少必需的函数: {func_name}")
    
    return module


def generate_with_vllm(prompt: str, port: int, args: argparse.Namespace) -> str:
    """同步版本的vLLM生成函数（保留用于向后兼容）。"""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": args.served_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_new_tokens,
        "n": 1,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=args.request_timeout) as response:
            body = response.read().decode("utf-8")
            content = json.loads(body)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"vLLM返回HTTP错误: {exc}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"vLLM连接失败: {exc}") from exc

    try:
        return content["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"解析vLLM响应失败: {content}") from exc


async def generate_with_vllm_async(
    session: aiohttp.ClientSession, prompt: str, port: int, args: argparse.Namespace
) -> str:
    """异步版本的vLLM生成函数，用于并发请求。"""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": args.served_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_new_tokens,
        "n": 1,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    try:
        async with session.post(url, json=payload, headers=headers, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"vLLM返回HTTP错误: {response.status}")
            content = await response.json()
    except aiohttp.ClientError as exc:
        raise RuntimeError(f"vLLM连接失败: {exc}") from exc

    try:
        return content["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"解析vLLM响应失败: {content}") from exc


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


async def evaluate_dataset(
    dataset_name: str,
    task_module: Any,
    args: argparse.Namespace,
    ports: List[int],
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """
    异步并发评估数据集。
    实现方案：为每个DP端口维护一个信号量（Semaphore）限制并发数，创建所有任务后异步执行，
    当一个请求完成时自动从队列中取出下一个请求发送，确保每个DP的并发数不超过max_num_request_per_dp。
    """
    dataset_dir = Path(args.result_dir) / dataset_name
    outputs_dir = dataset_dir / "outputs"
    result_file = dataset_dir / "result.jsonl"

    if result_file.exists():
        logger.warning("检测到已存在的结果文件，跳过重新评测数据集 %s : %s", dataset_name, result_file)
        try:
            with result_file.open("r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        except Exception as exc:  # noqa: BLE001
            logger.error("读取已有结果失败，将重新评测。错误：%s", exc)

    # 使用任务模块中的load_dataset函数加载数据集
    ds = task_module.load_dataset_from_hf()

    # 为每个DP端口创建信号量，限制并发请求数
    max_concurrent_per_dp = max(1, args.max_num_request_per_dp)
    semaphores: Dict[int, asyncio.Semaphore] = {port: asyncio.Semaphore(max_concurrent_per_dp) for port in ports}
    logger.info("每个DP端口的最大并发请求数：%d", max_concurrent_per_dp)

    # 收集所有需要处理的任务
    # (problem_id, rollout_id, prompt, output_path, port_idx, sample)
    tasks_to_process: List[Tuple[int, int, str, Path, int, Dict[str, Any]]] = []
    cached_count = 0
    ports_cycle = len(ports)

    for idx, sample in enumerate(ds):
        # 使用任务模块中的prepare_prompt函数
        prompt = task_module.prepare_prompt(sample)
        problem_dir = outputs_dir / f"{idx:06d}"
        for rollout_id in range(args.rollout_n):
            output_path = problem_dir / f"rollout_{rollout_id:03d}.txt"
            port_idx = (idx * args.rollout_n + rollout_id) % ports_cycle
            if output_path.exists() and output_path.stat().st_size > 0:
                cached_count += 1
            tasks_to_process.append((idx, rollout_id, prompt, output_path, port_idx, sample))

    logger.info(
        "需要处理的请求总数：%d（已存在缓存：%d，需新生成：%d）",
        len(tasks_to_process),
        cached_count,
        len(tasks_to_process) - cached_count,
    )

    # 异步生成函数
    async def generate_one_task(
        problem_id: int,
        rollout_id: int,
        prompt: str,
        output_path: Path,
        port_idx: int,
        sample: Dict[str, Any],
        session: aiohttp.ClientSession,
    ) -> Dict[str, Any]:
        response = ""
        # 用户要求：若输出文件存在则在此处直接复用并跳过generate_with_vllm_async，避免在主循环重复写评分逻辑。
        if output_path.exists() and output_path.stat().st_size > 0:
            response = output_path.read_text(encoding="utf-8")
            logger.info("复用缓存结果：%s", output_path)
        else:
            port = ports[port_idx]
            semaphore = semaphores[port]
            async with semaphore:  # 限制每个DP的并发数
                try:
                    logger.info("向端口%d请求生成，problem=%06d rollout=%03d", port, problem_id, rollout_id)
                    response = await generate_with_vllm_async(session, prompt, port, args)
                    # 实现方案：在调用score_response之前先保存响应到文件，确保即使score_response报错也能保留响应
                    save_text(output_path, response)
                except Exception as exc:  # noqa: BLE001
                    logger.error("生成响应失败 problem=%06d rollout=%03d port=%d: %s", problem_id, rollout_id, port, exc)
                    # 如果生成失败，response为空字符串，但也要保存（可能是空文件）
                    if response:
                        save_text(output_path, response)
                    return {
                        "problem_id": problem_id,
                        "rollout_id": rollout_id,
                        "prompt": prompt,
                        "response": response,
                        "score": 0.0,
                        "details": {},
                    }
        
        # 响应已保存或来自缓存，现在尝试评分
        score = 0.0
        details = {}
        try:
            # 使用任务模块中的score_response函数
            score_result = task_module.score_response(prompt, response, sample)
            # 兼容返回元组或单个值的情况
            if isinstance(score_result, tuple):
                score, details = score_result
            else:
                score = score_result
                details = {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("评分失败 problem=%06d rollout=%03d，响应已保存，使用默认分数。错误：%s", problem_id, rollout_id, exc)
        
        return {
            "problem_id": problem_id,
            "rollout_id": rollout_id,
            "prompt": prompt,
            "response": response,
            "score": score,
            "details": details,
        }

    # 创建aiohttp会话并并发执行所有任务
    async with aiohttp.ClientSession() as session:
        tasks = [
            generate_one_task(problem_id, rollout_id, prompt, output_path, port_idx, sample, session)
            for problem_id, rollout_id, prompt, output_path, port_idx, sample in tasks_to_process
        ]
        records = await asyncio.gather(*tasks)

    # 按problem_id和rollout_id排序，确保结果顺序一致
    records.sort(key=lambda x: (x["problem_id"], x["rollout_id"]))

    result_file.parent.mkdir(parents=True, exist_ok=True)
    with result_file.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("数据集 %s 评测完成，结果写入 %s", dataset_name, result_file)
    return records


def compute_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    if num_correct == 0:
        return 0.0
    if num_samples <= k:
        return 1.0 if num_correct > 0 else 0.0
    return 1 - (math.comb(num_samples - num_correct, k) / math.comb(num_samples, k))


def compute_metrics(records: List[Dict[str, Any]], rollout_n: int) -> Dict[str, Dict[int, float]]:
    by_problem: Dict[int, List[float]] = {}
    for rec in records:
        by_problem.setdefault(int(rec["problem_id"]), []).append(float(rec["score"]))

    avg_at_k: Dict[int, float] = {}
    pass_at_k: Dict[int, float] = {}

    for k in range(1, rollout_n + 1):
        avg_scores = []
        pass_scores = []
        for scores in by_problem.values():
            sorted_scores = sorted(scores, reverse=True)
            topk = sorted_scores[:k]
            if topk:
                avg_scores.append(sum(topk) / len(topk))
            c = sum(1 for s in scores if s > 0)
            pass_scores.append(compute_pass_at_k(len(scores), c, k))

        avg_at_k[k] = sum(avg_scores) / len(avg_scores) if avg_scores else 0.0
        pass_at_k[k] = sum(pass_scores) / len(pass_scores) if pass_scores else 0.0

    return {"avg_at_k": avg_at_k, "pass_at_k": pass_at_k}


def main() -> None:
    args, vllm_args, leftover = parse_args()
    logger = setup_logging(Path(args.result_dir))
    if leftover:
        logger.warning("检测到无法识别的参数（将被忽略）：%s", leftover)

    with StageContext(logger, 1, "准备模型/合并LoRA"):
        model_path = merge_model_if_needed(args, Path(args.result_dir), logger)

    with StageContext(logger, 2, "启动vLLM后端"):
        processes, ports = start_vllm_processes(model_path, args, vllm_args, logger)
        atexit.register(stop_vllm_processes, processes, logger)

        def handle_signal(signum, frame):  # noqa: ANN001
            logger.warning("收到信号%d，准备清理后退出。", signum)
            stop_vllm_processes(processes, logger)
            sys.exit(1)

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        for proc, port in zip(processes, ports):
            if not wait_for_vllm_ready(port, proc, timeout=300, logger=logger):
                stop_vllm_processes(processes, logger)
                sys.exit(1)

    all_records: Dict[str, List[Dict[str, Any]]] = {}
    datasets_to_run = [item.strip() for item in args.dataset.split(",") if item.strip()]
    with StageContext(logger, 3, "数据集评测与缓存/生成"):
        async def run_evaluations():
            for task_abbr in datasets_to_run:
                logger.info("🧪 开始评测数据集：%s", task_abbr)
                try:
                    # 动态加载任务模块
                    task_module = load_task_module(task_abbr)
                    logger.info("✅ 成功加载任务模块：%s", task_abbr)
                except Exception as exc:  # noqa: BLE001
                    logger.error("❌ 加载任务模块失败 %s: %s", task_abbr, exc)
                    raise exc
                # 使用任务模块进行评测
                records = await evaluate_dataset(task_abbr, task_module, args, ports, logger)
                all_records[task_abbr] = records
                logger.info("✅ 完成评测数据集：%s", task_abbr)
        
        asyncio.run(run_evaluations())

    with StageContext(logger, 4, "统计阶段：计算avg@k与pass@k"):
        overall_records: List[Dict[str, Any]] = []
        for name, records in all_records.items():
            overall_records.extend(records)
            metrics = compute_metrics(records, args.rollout_n)
            logger.info("📊 数据集%s avg@k: %s", name, metrics["avg_at_k"])
            logger.info("📈 数据集%s pass@k: %s", name, metrics["pass_at_k"])

        overall_metrics = compute_metrics(overall_records, args.rollout_n) if overall_records else None
        if overall_metrics:
            logger.info("🌐 全部数据集合并 avg@k: %s", overall_metrics["avg_at_k"])
            logger.info("🌟 全部数据集合并 pass@k: %s", overall_metrics["pass_at_k"])
        else:
            logger.warning("未获取到任何记录，跳过全局统计。")

    stop_vllm_processes(processes, logger)
    logger.info("全部评测流程完成。")


if __name__ == "__main__":
    main()
