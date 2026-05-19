from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .task import discover_task_names, resolve_task_file

def _parse_optional_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value for --enable-thinking: {value}. Use true/false."
    )

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NanoEval pipeline: step01 prepare inputs, step02 inference."
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["step01", "step02", "step03", "all"],
        default="step01",
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help=(
            "Comma-separated task names. Each task can optionally specify pass-k as task@k "
            "(for example: aime2024@4,aime2025@8). Use 'all' to load all *.jsonl under task directory."
        ),
    )
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=Path("outputs/nano_eval"),
        help="Directory that stores task jsonl files.",
    )
    parser.add_argument(
        "--pass-k",
        type=int,
        default=1,
        help="Default repeated attempts per question. Used when a task in --tasks does not specify @k.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Working directory for generated artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for step01 prepared input jsonl.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input jsonl for step02 inference. If omitted, uses --output.",
    )
    parser.add_argument(
        "--inference-output",
        type=Path,
        required=True,
        help="Output jsonl for step02 inference results.",
    )
    parser.add_argument(
        "--score-output",
        type=Path,
        required=True,
        help="Output jsonl for per-instance judged results in step03.",
    )
    parser.add_argument(
        "--final-eval-output",
        type=Path,
        required=True,
        help="Output jsonl for aggregated metrics in step03.",
    )
    parser.add_argument(
        "--eval-input",
        type=Path,
        default=None,
        help="Input jsonl for step03 evaluation. If omitted, uses --inference-output.",
    )
    parser.add_argument(
        "--n-proc",
        type=int,
        default=32,
        help="Number of processes for reward judging in step03.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["mock", "offline", "online", "online_ray"],
        default="mock",
        help="Inference backend for step02.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Local model path for offline backend.",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size for offline backend.",
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=1,
        help="Data parallel size for offline backend.",
    )
    parser.add_argument(
        "--chat-template-model-path",
        type=str,
        default=None,
        help="Model path used to apply chat template in step01. Defaults to --model-path.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        nargs="?",
        const="",
        default=None,
        help=(
            "System prompt used when applying chat template in step01. "
            "Use --system-prompt 'text' to add system message; "
            "use --system-prompt (empty) to add empty system message; "
            "omit to skip system message entirely."
        ),
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for online backend.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="API base url for online backend.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for online backend.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="Online backend concurrency.",
    )
    parser.add_argument(
        "--ray-num-actors",
        type=int,
        default=None,
        help="Optional actor count for online_ray backend. If unset, derived from --concurrency.",
    )
    parser.add_argument(
        "--ray-worker-concurrency",
        type=int,
        default=None,
        help="Optional per-actor async concurrency for online_ray backend. If unset, derived from --concurrency.",
    )
    parser.add_argument(
        "--online-request-timeout-s",
        type=float,
        default=120.0,
        help="Per-request timeout seconds for online/online_ray backend network calls.",
    )
    parser.add_argument(
        "--online-stall-log-interval-s",
        type=float,
        default=15.0,
        help="Heartbeat interval seconds when online_ray has no completed chunks.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for inference.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Sampling max tokens for inference.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Optional nucleus sampling top-p for inference.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional top-k sampling value for inference.",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=None,
        help="Optional minimum probability threshold for sampling.",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=None,
        help="Optional presence penalty for token generation.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Optional repetition penalty for token generation.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["low", "medium", "high"],
        default=None,
        help="Optional reasoning effort level for models/APIs that support it.",
    )
    parser.add_argument(
        "--enable-thinking",
        nargs="?",
        const=True,
        default=None,
        type=_parse_optional_bool,
        help=(
            "Set chat template thinking for online-style backends when supported. "
            "Use --enable-thinking (true), or --enable-thinking false."
        ),
    )
    return parser


def parse_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    return parser.parse_args(argv)


def parse_task_names(tasks_arg: str, task_dir: Path) -> List[str]:
    normalized_arg = tasks_arg.strip()
    if not normalized_arg:
        raise ValueError("--tasks cannot be empty.")

    if normalized_arg.lower() == "all":
        names = discover_task_names(task_dir)
        if not names:
            raise ValueError(f"No task jsonl found under: {task_dir}")
        return names

    task_specs = _parse_task_specs(normalized_arg)
    names = [task_name for task_name, _ in task_specs]
    _validate_task_names(names=names, task_dir=task_dir)
    return names


def _parse_task_specs(tasks_arg: str) -> List[Tuple[str, str | None]]:
    task_specs: List[Tuple[str, str | None]] = []
    for task_spec in [item.strip() for item in tasks_arg.split(",") if item.strip()]:
        if "@" in task_spec:
            task_part, pass_k_part = task_spec.split("@", 1)
            task_name = task_part.strip()
            pass_k_value = pass_k_part.strip()
        else:
            task_name = task_spec
            pass_k_value = None
        if task_name.endswith(".jsonl"):
            task_name = task_name[:-6]
        task_name = task_name.strip()
        if not task_name:
            raise ValueError(f"Invalid task spec in --tasks: {task_spec}")
        task_specs.append((task_name, pass_k_value))
    return task_specs


def _validate_task_names(names: List[str], task_dir: Path) -> None:
    if not names:
        raise ValueError("--tasks resolved to an empty list.")

    missing_names: List[str] = []
    for name in names:
        try:
            task_file = resolve_task_file(task_name=name, task_dir=task_dir)
        except ValueError:
            missing_names.append(name)
            continue
        if not task_file.exists():
            missing_names.append(name)
    if missing_names:
        raise ValueError(
            f"Missing task files under {task_dir}: {', '.join(sorted(missing_names))}"
        )


def parse_task_pass_k(
    tasks_arg: str,
    task_dir: Path,
    default_pass_k: int,
) -> Tuple[List[str], Dict[str, int]]:
    if default_pass_k <= 0:
        raise ValueError("--pass-k must be a positive integer.")

    normalized_arg = tasks_arg.strip()
    if not normalized_arg:
        raise ValueError("--tasks cannot be empty.")

    if normalized_arg.lower() == "all":
        task_names = parse_task_names(tasks_arg=tasks_arg, task_dir=task_dir)
        pass_k_by_task: Dict[str, int] = {task_name: default_pass_k for task_name in task_names}
        return task_names, pass_k_by_task

    task_specs = _parse_task_specs(normalized_arg)
    task_names = [task_name for task_name, _ in task_specs]
    _validate_task_names(names=task_names, task_dir=task_dir)
    pass_k_by_task: Dict[str, int] = {task_name: default_pass_k for task_name in task_names}

    for task_name, pass_k_text in task_specs:
        if pass_k_text is None:
            continue
        if task_name not in pass_k_by_task:
            raise ValueError(f"Unknown task in --tasks: {task_name}")
        if not pass_k_text.isdigit():
            raise ValueError(
                f"Invalid pass-k value in --tasks spec '{task_name}@{pass_k_text}'. "
                "Expected a positive integer after '@'."
            )
        pass_k_value = int(pass_k_text)
        if pass_k_value <= 0:
            raise ValueError(
                f"Invalid pass-k value in --tasks spec '{task_name}@{pass_k_text}'. "
                "pass-k must be a positive integer."
            )
        pass_k_by_task[task_name] = pass_k_value

    return task_names, pass_k_by_task
