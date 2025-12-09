import argparse
import asyncio
import atexit
import logging
import shutil
import signal
import sys
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
from datasets import load_dataset
import aiohttp

from perl.eval.utils import StageContext, setup_logging, merge_model_if_needed
from perl.eval.vllm import (
    extract_vllm_args,
    start_vllm_processes,
    stop_vllm_processes,
    wait_for_vllm_ready,
    generate_with_vllm_async,
)
from perl.eval.grader import grade_answer_perl


PROMPT_TEMPLATES = {
    "lighteval": """{problem} Please reason step by step, and put your final answer within \\boxed{{}}.""",
    "open-r1": """
Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.

{problem}

Remember to put your answer on its own line after "Answer:".
""".strip(),
}

DATASETS = {
    "aime2024": ("HuggingFaceH4/aime_2024", "train"),
    "aime2025": ("yentinglin/aime_2025", "train"),
    "amc2023": ("zwhe99/amc23", "test"),
    "math500": ("HuggingFaceH4/MATH-500", "test"),
    "minerva": ("math-ai/minervamath", "test"),
    "hmmt2025": ("FlagEval/HMMT_2025", "train"),
}


def load_dataset_from_hf(dataset_name: str):
    if dataset_name in DATASETS:
        hf_name, split = DATASETS[dataset_name]
        return load_dataset(hf_name, split=split)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def prepare_prompt(
    dataset_name: str, sample: Dict[str, Any], prompt_template: str
) -> str:
    """Construct model input prompt based on sample, modify as needed."""
    problem = None
    if "problem" in sample:
        problem = sample["problem"]
    elif "question" in sample:
        problem = sample["question"]
    elif "prompt" in sample:
        problem = sample["prompt"]
    else:
        raise ValueError(f"Unsupported sample format: {sample}")
    return prompt_template.format(problem=problem)


def score_response(
    dataset_name: str, response: str, sample: Dict[str, Any]
) -> Tuple[float, float]:
    """
    Returns:
      - score: float, score of the response
      - format_score: float, score of the response format
    """
    ground_truth = None
    if "answer" in sample:
        ground_truth = sample["answer"]
    elif "label" in sample:
        ground_truth = sample["label"]
    else:
        raise ValueError(f"Unsupported sample format: {sample}")
    return grade_answer_perl(response, str(ground_truth))


def parse_args() -> Tuple[argparse.Namespace, List[str], List[str]]:
    parser = argparse.ArgumentParser(
        description="Evaluation entry script, supports model merging, vLLM startup, and multi-dataset evaluation."
    )
    parser.add_argument(
        "--result-dir",
        required=True,
        help="Directory for intermediate processes and result output.",
    )
    parser.add_argument("--model", required=True, help="Base model name or path.")
    parser.add_argument(
        "--adapter",
        default="",
        help="LoRA/PEFT adapter path, leave empty for no merge.",
    )
    parser.add_argument(
        "--dataset",
        default="aime2024",
        help="Dataset abbreviation to evaluate, comma separated (e.g., aime2024).",
    )
    parser.add_argument(
        "--prompt-format",
        default="lighteval",
        help="Prompt format template to use.",
    )
    parser.add_argument(
        "--rollout-n",
        type=int,
        default=1,
        help="Number of rollouts to generate per sample.",
    )
    parser.add_argument(
        "--serve-port", type=int, default=8000, help="First vLLM backend port number."
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=1,
        help="Number of data parallel backends (start multiple vLLMs).",
    )
    parser.add_argument(
        "--tp-size", type=int, default=1, help="Tensor parallel size passed to vLLM."
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Verify needed GPU count before running, error if insufficient.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help="GPU memory utilization limit passed to vLLM (0~1), controls memory usage per card.",
    )
    parser.add_argument(
        "--seed", type=float, default=None, help="Generation random seed."
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Generation temperature."
    )
    parser.add_argument("--top-p", type=float, default=1.0, help="Generation top-p.")
    parser.add_argument(
        "--max-new-tokens", type=int, default=131072, help="Generation length."
    )
    parser.add_argument(
        "--dtype", default="auto", help="Model dtype, used during merging."
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="Whether to trust remote code."
    )
    parser.add_argument(
        "--served-model-name", default="eval-model", help="Model name exposed by vLLM."
    )
    parser.add_argument(
        "--api-key", default="dummy", help="API Key for OpenAI compatible interface."
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=3600.0,
        help="Timeout for a single request.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="For debugging, limit number of evaluation samples.",
    )
    parser.add_argument(
        "--max-num-request",
        type=int,
        default=None,
        help="Max number of concurrent requests per data parallel (DP) vLLM backend.",
    )

    args, unknown = parser.parse_known_args()

    if args.max_num_request is None:
        args.max_num_request = args.dp_size
    else:
        assert args.max_num_request > 0
        assert args.max_num_request % args.dp_size == 0, (
            f"args.max_num_request({args.max_num_request}) must be divisible by args.dp_size({args.dp_size})"
        )

    vllm_args, leftover = extract_vllm_args(unknown)
    return args, vllm_args, leftover


class ProgressVisualizer:
    def __init__(
        self,
        filepath: Path,
        problem_n: int,
        rollout_n: int,
        completed: Set[Tuple[int, int]],
    ) -> None:
        self.filepath = filepath
        self.problem_n = problem_n
        self.rollout_n = rollout_n
        # Row: rollout_id, Col: problem_id
        self.grid = [["." for _ in range(problem_n)] for _ in range(rollout_n)]
        for pid, rid in completed:
            if 0 <= rid < rollout_n and 0 <= pid < problem_n:
                self.grid[rid][pid] = "X"
        self.lock = asyncio.Lock()
        self._write_sync()

    def _write_sync(self) -> None:
        try:
            with self.filepath.open("w", encoding="utf-8") as f:
                for row in self.grid:
                    f.write("".join(row) + "\n")
        except Exception:
            pass

    async def update(self, problem_id: int, rollout_id: int) -> None:
        if 0 <= rollout_id < self.rollout_n and 0 <= problem_id < self.problem_n:
            async with self.lock:
                if self.grid[rollout_id][problem_id] != "X":
                    self.grid[rollout_id][problem_id] = "X"
                    await asyncio.get_running_loop().run_in_executor(
                        None, self._write_sync
                    )

    def cleanup(self) -> None:
        try:
            if self.filepath.exists():
                self.filepath.unlink()
        except Exception:
            pass


async def generate_responses(
    args: argparse.Namespace,
    dataset_name: str,
    rollout_n: int,
    ports: List[int],
    logger: logging.Logger,
    semaphores: Dict[int, asyncio.Semaphore],
) -> None:
    """
    Asynchronously generate responses and save to output.jsonl.
    Implementation: Read existing output.jsonl to build cache, only generate missing entries.
    Generated results are appended to output.jsonl in real-time.
    """
    dataset_dir = Path(args.result_dir) / dataset_name
    output_file = dataset_dir / "output.jsonl"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    with StageContext(logger, f"C.1[{dataset_name}]", "Reading cached output"):
        generated_results: List[Dict[str, Any]] = []
        cache: Set[Tuple[int, int]] = set()

        if output_file.exists():
            with output_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if (
                            "problem_id" in data
                            and "rollout_id" in data
                            and "response" in data
                            and data["response"] != ""
                            and int(data["rollout_id"]) < rollout_n
                        ):
                            generated_results.append(data)
                            cache.add((data["problem_id"], data["rollout_id"]))
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON line in output.jsonl, skipped.")

        logger.info("Loaded cache entries: %d", len(generated_results))

    with StageContext(logger, f"C.2[{dataset_name}]", "Preparing generation tasks"):
        ds = load_dataset_from_hf(dataset_name)
        # max_concurrent_per_dp and semaphores are now handled externally and passed in

        tasks_to_process: List[Tuple[int, int, str, int]] = []
        ports_cycle = len(ports)

        prompt_template = PROMPT_TEMPLATES[args.prompt_format]

        for idx, sample in enumerate(ds):
            prompt = prepare_prompt(dataset_name, sample, prompt_template)
            for rollout_id in range(rollout_n):
                if (idx, rollout_id) in cache:
                    continue
                # port_idx = idx % ports_cycle
                port_idx = (idx * rollout_n + rollout_id) % ports_cycle
                tasks_to_process.append((idx, rollout_id, prompt, port_idx))

        logger.info("New requests to generate: %d", len(tasks_to_process))

        visualizer = ProgressVisualizer(
            dataset_dir / "process.txt", len(ds), rollout_n, cache
        )

        if not tasks_to_process:
            logger.info("All requests exist in cache, no generation needed.")
            visualizer.cleanup()
            return

    with StageContext(logger, f"C.3[{dataset_name}]", "Parallel Generation"):
        file_lock = asyncio.Lock()

        async def generate_one_task(
            problem_id: int,
            rollout_id: int,
            prompt: str,
            port_idx: int,
            session: aiohttp.ClientSession,
        ) -> None:
            port = ports[port_idx]
            semaphore = semaphores[port]
            response = ""

            async with semaphore:
                try:
                    response = await generate_with_vllm_async(
                        session, prompt, port, args
                    )
                except Exception as exc:
                    logger.error(
                        "Generation failed problem=%06d rollout=%03d port=%d: %s",
                        problem_id,
                        rollout_id,
                        port,
                        exc,
                    )
                    return

            record = {
                "problem_id": problem_id,
                "rollout_id": rollout_id,
                "response": response,
            }

            generated_results.append(record)

            async with file_lock:
                with output_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            await visualizer.update(problem_id, rollout_id)

        connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                generate_one_task(pid, rid, pmt, pidx, session)
                for pid, rid, pmt, pidx in tasks_to_process
            ]
            await asyncio.gather(*tasks)
            visualizer.cleanup()

        logger.info(
            "Dataset %s generation complete, results saved to %s",
            dataset_name,
            output_file,
        )


def evaluate_dataset_results(
    args: argparse.Namespace,
    dataset_name: str,
    rollout_n: int,
    logger: logging.Logger,
) -> Dict[str, Dict[int, float]]:
    """
    Evaluation stage: Read output.jsonl, score and generate result.jsonl, return stats metrics.
    """
    dataset_dir = Path(args.result_dir) / dataset_name
    output_file = dataset_dir / "output.jsonl"
    result_file = dataset_dir / "result.jsonl"
    result_json_file = dataset_dir / "result.json"

    with StageContext(logger, f"D.1[{dataset_name}]", "Loading model output"):
        if not output_file.exists():
            raise ValueError(f"output.jsonl not found, cannot evaluate: {dataset_name}")

        outputs_map: Dict[int, List[Tuple[int, str]]] = {}
        with output_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if "problem_id" in d and "rollout_id" in d:
                        outputs_map.setdefault(d["problem_id"], []).append(
                            (d["rollout_id"], d.get("response", ""))
                        )
                except json.JSONDecodeError:
                    pass

    with StageContext(logger, f"D.2[{dataset_name}]", "Loading original dataset"):
        ds = load_dataset_from_hf(dataset_name)

    with StageContext(logger, f"D.3[{dataset_name}]", "Parallel Evaluation & Metrics"):
        records_for_metrics: List[Dict[str, Any]] = []
        raw_stats_list: List[Dict[str, Any]] = []

        prompt_template = PROMPT_TEMPLATES[args.prompt_format]
        with result_file.open("w", encoding="utf-8") as rf:
            for idx, sample in enumerate(ds):
                problem_id = idx
                prompt = prepare_prompt(dataset_name, sample, prompt_template)

                rollouts = outputs_map.get(problem_id, [])
                # Sort by rollout_id
                rollouts.sort(key=lambda x: x[0])
                rollout_dict = {r[0]: r[1] for r in rollouts}

                responses = []
                scores = []
                format_scores = []

                for rid in range(rollout_n):
                    if rid not in rollout_dict:
                        raise ValueError(
                            f"Missing result: problem_id={problem_id} rollout_id={rid}. Please check if generation requests failed."
                        )
                    resp = rollout_dict.get(rid, "")
                    responses.append(resp)

                    if resp:
                        score, format_score = score_response(dataset_name, resp, sample)
                    else:
                        score, format_score = 0.0, 0.0
                    scores.append(score)
                    format_scores.append(format_score)

                    records_for_metrics.append(
                        {"problem_id": problem_id, "rollout_id": rid, "score": score}
                    )

                if scores:
                    avg_val = statistics.mean(scores)
                    max_val = max(scores)
                    min_val = min(scores)
                    try:
                        std_val = statistics.stdev(scores)
                    except statistics.StatisticsError:
                        std_val = 0.0
                else:
                    avg_val = max_val = min_val = std_val = 0.0

                format_score_avg = (
                    statistics.mean(format_scores) if format_scores else 0.0
                )

                record = {
                    "problem_id": problem_id,
                    "prompt": prompt,
                    "responses": responses,
                    "scores": scores,
                    "avg": avg_val,
                    "max": max_val,
                    "min": min_val,
                    "std": std_val,
                    "format_score_avg": format_score_avg,
                    "data_source": dataset_name,
                }
                rf.write(json.dumps(record, ensure_ascii=False) + "\n")

                raw_stats_list.append(
                    {
                        "problem_id": problem_id,
                        "avg": avg_val,
                        "max": max_val,
                        "min": min_val,
                        "std": std_val,
                        "format_score_avg": format_score_avg,
                    }
                )

    with StageContext(logger, f"D.4[{dataset_name}]", "Summarizing and writing files"):
        if raw_stats_list:
            summary = {
                "avg": statistics.mean(x["avg"] for x in raw_stats_list),
                "max": statistics.mean(x["max"] for x in raw_stats_list),
                "min": statistics.mean(x["min"] for x in raw_stats_list),
                "std": statistics.mean(x["std"] for x in raw_stats_list),
                "format_score_avg": statistics.mean(
                    x["format_score_avg"] for x in raw_stats_list
                ),
            }
        else:
            summary = {
                "avg": 0.0,
                "max": 0.0,
                "min": 0.0,
                "std": 0.0,
                "format_score_avg": 0.0,
            }

        final_json = {
            "data_source": dataset_name,
            "rollout_n": rollout_n,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": summary,
            "raw": raw_stats_list,
            "response_example": [
                outputs_map[0][0] if outputs_map else [],
            ],
        }

        with result_json_file.open("w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=2, ensure_ascii=False)

        logger.info(
            "Evaluation complete, results written to %s and %s",
            result_file,
            result_json_file,
        )


async def main() -> None:
    args, vllm_args, leftover = parse_args()
    logger = setup_logging(Path(args.result_dir))
    if leftover:
        logger.warning(
            "Detected unrecognized arguments (will be ignored): %s", leftover
        )

    with StageContext(logger, "A", "Prepare Model/Merge LoRA"):
        model_path = merge_model_if_needed(args, Path(args.result_dir), logger)

    with StageContext(logger, "B", "Start vLLM Backends"):
        processes, ports = start_vllm_processes(model_path, args, vllm_args, logger)
        atexit.register(stop_vllm_processes, processes, logger)

        def handle_signal(signum, frame):  # noqa: ANN001
            logger.warning(
                "Received signal %d, preparing to clean up and exit.", signum
            )
            stop_vllm_processes(processes, logger)
            sys.exit(1)

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        for proc, port in zip(processes, ports):
            if not wait_for_vllm_ready(port, proc, timeout=300, logger=logger):
                stop_vllm_processes(processes, logger)
                sys.exit(1)

    # Initialize global semaphores
    dp_size = max(1, args.dp_size)
    max_concurrent_per_dp = max(1, args.max_num_request // dp_size)
    semaphores = {port: asyncio.Semaphore(max_concurrent_per_dp) for port in ports}
    logger.info(
        "Global concurrency control initialized: Max concurrency per DP process=%d",
        max_concurrent_per_dp,
    )

    async def process_dataset_task(
        args: argparse.Namespace,
        dataset_name: str,
        rollout_n: int,
        ports: List[int],
        logger: logging.Logger,
        semaphores: Dict[int, asyncio.Semaphore],
    ) -> None:
        with StageContext(
            logger, f"C[{dataset_name}]", "Dataset Generation (Cache/Gen)"
        ):
            await generate_responses(
                args, dataset_name, rollout_n, ports, logger, semaphores
            )

        with StageContext(logger, f"D[{dataset_name}]", "Evaluation & Statistics"):
            # evaluate_dataset_results is synchronous CPU-bound task, put in thread pool to avoid blocking other concurrent tasks
            await asyncio.to_thread(
                evaluate_dataset_results, args, dataset_name, rollout_n, logger
            )

    datasets_to_run = [item.strip() for item in args.dataset.split(",") if item.strip()]
    tasks = []

    for task_abbr in datasets_to_run:
        if "@" in task_abbr:
            dataset_name = task_abbr.split("@")[0]
            rollout_n = int(task_abbr.split("@")[1])
        else:
            dataset_name = task_abbr
            rollout_n = args.rollout_n

        tasks.append(
            asyncio.create_task(
                process_dataset_task(
                    args, dataset_name, rollout_n, ports, logger, semaphores
                )
            )
        )

    if tasks:
        logger.info(
            "Submitted %d dataset tasks concurrently, starting execution...", len(tasks)
        )
        await asyncio.gather(*tasks)
    else:
        logger.warning("No dataset tasks to execute.")

    stop_vllm_processes(processes, logger)
    logger.info("All evaluation processes completed.")

    if args.adapter:
        merged_model_dir = Path(args.result_dir) / "model"
        if merged_model_dir.exists():
            logger.info("Deleting merged model directory: %s", merged_model_dir)
            try:
                shutil.rmtree(merged_model_dir)
                logger.info("Merged model directory deleted.")
            except Exception as e:
                logger.warning("Failed to delete merged model directory: %s", e)


if __name__ == "__main__":
    asyncio.run(main())
