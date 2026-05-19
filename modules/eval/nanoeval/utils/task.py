from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence


TASK_TO_JSONL: Dict[str, str] = {
    "aime2024": "aime2024.jsonl",
    "aime2025": "aime2025.jsonl",
    "aime2026": "aime2026.jsonl",
    "beyond_aime": "beyond_aime.jsonl",
    "amc2023": "amc2023.jsonl",
    "math500": "math500.jsonl",
    "minerva": "minerva.jsonl",
    "hmmt2025": "hmmt2025.jsonl",
    "gpqa_diamond": "gpqa_diamond.jsonl",
    "mmlu": "mmlu.jsonl",
    "mmlu_pro": "mmlu_pro.jsonl",
    "mmlu_prox": "mmlu_prox.jsonl",
    "ceval": "ceval.jsonl",
    "ifeval": "ifeval.jsonl",
    "ifbench": "ifbench.jsonl",
}


def resolve_task_file(task_name: str, task_dir: Path) -> Path:
    filename = TASK_TO_JSONL.get(task_name)
    if filename is None:
        raise ValueError(f"Unsupported task name: {task_name}")
    return task_dir / filename


def discover_task_names(task_dir: Path) -> List[str]:
    available_tasks: List[str] = []
    for task_name in TASK_TO_JSONL:
        if resolve_task_file(task_name=task_name, task_dir=task_dir).exists():
            available_tasks.append(task_name)
    return sorted(available_tasks)


def load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Task file does not exist: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Invalid JSON object at {path}:{line_number}. Each line must be a JSON object."
                )
            rows.append(payload)
    return rows


def expand_records_for_pass_k(
    task_name: str,
    rows: Iterable[Dict[str, Any]],
    pass_k: int,
    prompt_transform: Callable[[str], str] | None = None,
) -> List[Dict[str, Any]]:
    if pass_k <= 0:
        raise ValueError("pass_k must be a positive integer.")

    expanded_rows: List[Dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        question_id = str(row.get("question_id") or row.get("id") or f"{task_name}_{row_index}")
        prompt_value = str(row.get("prompt", ""))
        if prompt_transform is not None:
            prompt_value = prompt_transform(prompt_value)
        for sample_index in range(pass_k):
            expanded_rows.append(
                {
                    **row,
                    "prompt": prompt_value,
                    "id": f"{question_id}_{sample_index}",
                    "question_id": question_id,
                    "source": task_name,
                    "sample_index": sample_index,
                }
            )
    return expanded_rows


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    record_count = 0
    with path.open("w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")
            record_count += 1
    return record_count


def prepare_eval_input(
    task_names: Sequence[str],
    task_dir: Path,
    pass_k_by_task: Dict[str, int],
    output_path: Path,
    chat_template_model_path: str | None = None,
    system_prompt: str | None = None,
) -> Dict[str, Any]:
    import logging
    logging.info("prepare_eval_input: system_prompt=%r", system_prompt)
    if not task_names:
        raise ValueError("No task names provided.")

    prompt_transform: Callable[[str], str] | None = None
    if chat_template_model_path is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            chat_template_model_path,
            trust_remote_code=True,
        )
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError(
                "Tokenizer does not support apply_chat_template. "
                f"model_path={chat_template_model_path}"
            )

        def _to_chat_prompt(user_prompt: str) -> str:
            messages: List[Dict[str, str]] = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        prompt_transform = _to_chat_prompt

    merged_records: List[Dict[str, Any]] = []
    task_sizes: Dict[str, int] = {}

    for task_name in task_names:
        current_pass_k = pass_k_by_task.get(task_name)
        if current_pass_k is None:
            raise ValueError(f"Missing pass-k for task: {task_name}")
        task_file = resolve_task_file(task_name=task_name, task_dir=task_dir)
        rows = load_jsonl_records(task_file)
        expanded_rows = expand_records_for_pass_k(
            task_name=task_name,
            rows=rows,
            pass_k=current_pass_k,
            prompt_transform=prompt_transform,
        )
        task_sizes[task_name] = len(expanded_rows)
        merged_records.extend(expanded_rows)

    total_count = write_jsonl(path=output_path, records=merged_records)
    return {
        "task_count": len(task_names),
        "instance_count": total_count,
        "pass_k_by_task": pass_k_by_task,
        "task_sizes": task_sizes,
        "output_path": str(output_path),
    }
