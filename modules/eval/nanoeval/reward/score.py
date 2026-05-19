import json
import logging
import csv
import re
from typing import Dict, List
from pathlib import Path
from multiprocessing import Pool

from .reward import judge_router

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
_THINK_BLOCK_PATTERN = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)
_FINAL_ANSWER_PATTERN = re.compile(r"\b(final\s+answer|answer)\s*[:：]\s*", flags=re.IGNORECASE)

def instance_judge(
    instance: Dict
) -> Dict:
    # ------------------ get the instance information ------------------
    response = str(instance.get("response", "") or "")
    response, extracted_thinking = _split_response_and_thinking(
        response=response,
        existing_thinking=instance.get("thinking"),
    )
    if extracted_thinking:
        existing_thinking = str(instance.get("thinking", "") or "").strip()
        if existing_thinking:
            if extracted_thinking not in existing_thinking:
                instance["thinking"] = f"{existing_thinking}\n\n{extracted_thinking}"
        else:
            instance["thinking"] = extracted_thinking
    instance["response"] = response
    label = instance.get("label", "")
    source = instance.get("source", None)
    
    other_kwargs = {
        k: v for k, v in instance.items() if k not in ["response", "label", "source"]
    }
    
    # ------------------ judge the instance ------------------
    judge_res = judge_router(
        response=response,
        label=label,
        source=source,
        **other_kwargs
    )
    
    # ------------------ update the instance ------------------
    instance.update(judge_res)
    return instance


def _split_response_and_thinking(response: str, existing_thinking: object = None) -> tuple[str, str]:
    """
    Split response into thinking and final response.

    Strategy:
    1. If </thinking> or <|end_of_thought|> exists: content before is thinking, after is response
    2. Otherwise: all content is response, thinking is empty (unless existing_thinking provided)
    """
    text = str(response or "").strip()
    if not text:
        return "", ""

    # Define split patterns: (pattern, include_pattern_in_thinking)
    split_patterns = [
        ("</thinking>", True),
        ("<|end_of_thought|>", True),
    ]

    for pattern, include_in_thinking in split_patterns:
        idx = text.find(pattern)
        if idx != -1:
            if include_in_thinking:
                thinking = text[:idx + len(pattern)].strip()
                response_text = text[idx + len(pattern):].strip()
            else:
                thinking = text[:idx].strip()
                response_text = text[idx + len(pattern):].strip()
            return response_text, thinking

    # No split pattern found - check for existing_thinking prefix
    if isinstance(existing_thinking, str):
        prefix = existing_thinking.strip()
        if prefix and text.startswith(prefix):
            return text[len(prefix):].lstrip(), prefix

    # No split pattern, no prefix match - all is response
    return text, ""


def _build_task_question_scores(items: List[Dict]) -> Dict[str, Dict[str, List[float]]]:
    grouped_scores: Dict[str, Dict[str, List[float]]] = {}
    for item in items:
        ds_name = item.get("source", "unknown")
        q_id = item.get("question_id", "unknown")
        score = 1.0 if item.get("pass", False) else 0.0
        grouped_scores.setdefault(ds_name, {}).setdefault(q_id, []).append(score)
    return grouped_scores


def _count_text_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_TOKEN_PATTERN.findall(text))


def _extract_answer_tokens(item: Dict) -> int:
    usage = item.get("usage", {})
    if isinstance(usage, dict):
        completion_tokens = usage.get("completion_tokens")
        if completion_tokens is None:
            completion_tokens = usage.get("output_tokens")
        if isinstance(completion_tokens, (int, float)) and completion_tokens >= 0:
            return int(completion_tokens)
    response_text = str(item.get("response", "") or "")
    thinking_text = str(item.get("thinking", "") or "")
    return _count_text_tokens(response_text) + _count_text_tokens(thinking_text)


def _compute_length_metrics(items: List[Dict]) -> Dict[str, float]:
    total_token_lengths: List[int] = []
    thinking_token_lengths: List[int] = []
    for item in items:
        thinking_text = str(item.get("thinking", "") or "")
        response_tokens = _extract_answer_tokens(item)
        thinking_tokens = _count_text_tokens(thinking_text)
        total_token_lengths.append(response_tokens)
        thinking_token_lengths.append(thinking_tokens)

    if not total_token_lengths:
        return {
            "avg_total_tokens": 0.0,
            "avg_thinking_tokens": 0.0,
            "max_thinking_tokens": 0.0,
            "min_thinking_tokens": 0.0,
        }

    return {
        "avg_total_tokens": sum(total_token_lengths) / len(total_token_lengths),
        "avg_thinking_tokens": sum(thinking_token_lengths) / len(thinking_token_lengths),
        "max_thinking_tokens": float(max(thinking_token_lengths)),
        "min_thinking_tokens": float(min(thinking_token_lengths)),
    }


def _calculate_metrics(
    grouped_scores: Dict[str, Dict[str, List[float]]],
    items: List[Dict],
) -> Dict[str, Dict[str, float]]:
    final_results = {}
    items_by_source: Dict[str, List[Dict]] = {}
    for item in items:
        ds_name = item.get("source", "unknown")
        items_by_source.setdefault(ds_name, []).append(item)

    for ds_name, q_map in grouped_scores.items():
        all_scores = []
        pass_at_k_scores = []

        for scores in q_map.values():
            all_scores.extend(scores)
            pass_at_k_scores.append(1.0 if any(s >= 1.0 for s in scores) else 0.0)

        final_results[ds_name] = {
            "avg_k": sum(all_scores) / len(all_scores) if all_scores else 0,
            "pass_k": sum(pass_at_k_scores) / len(pass_at_k_scores) if pass_at_k_scores else 0,
        }
        final_results[ds_name].update(_compute_length_metrics(items_by_source.get(ds_name, [])))

    # Overall metrics across all datasets
    overall_all_scores = []
    overall_pass_at_k_scores = []
    for q_map in grouped_scores.values():
        for scores in q_map.values():
            overall_all_scores.extend(scores)
            overall_pass_at_k_scores.append(1.0 if any(s >= 1.0 for s in scores) else 0.0)

    final_results["overall"] = {
        "avg_k": sum(overall_all_scores) / len(overall_all_scores) if overall_all_scores else 0,
        "pass_k": sum(overall_pass_at_k_scores) / len(overall_pass_at_k_scores) if overall_pass_at_k_scores else 0,
    }
    final_results["overall"].update(_compute_length_metrics(items))

    return final_results


def _build_pass_at_k_flags(
    grouped_scores: Dict[str, Dict[str, List[float]]]
) -> Dict[tuple[str, str], bool]:
    flags: Dict[tuple[str, str], bool] = {}
    for ds_name, q_map in grouped_scores.items():
        for question_id, scores in q_map.items():
            flags[(ds_name, question_id)] = any(score >= 1.0 for score in scores)
    return flags

def eval_results(
    eval_output_file: Path,
    score_output_file: Path,
    final_eval_output_file: Path,
    final_eval_csv_output_file: Path | None = None,
    n_proc: int = 32
) -> Dict[str, Dict[str, float]]:
    
    logging.info(f"Scoring eval results from {eval_output_file} (num_proc={n_proc})...")
    
    items: List[Dict] = []
    with open(eval_output_file, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                items.append(payload)
    logging.info(f"Loaded {len(items)} records; running judge_router...")

    if n_proc and n_proc > 1:
        with Pool(processes=n_proc) as pool:
            items = pool.map(instance_judge, items)
    else:
        items = [instance_judge(item) for item in items]
    logging.info("Judging complete; computing metrics...")
    grouped_scores = _build_task_question_scores(items)
    pass_at_k_flags = _build_pass_at_k_flags(grouped_scores)
    for item in items:
        key = (item.get("source", "unknown"), item.get("question_id", "unknown"))
        item["pass_at_k"] = pass_at_k_flags.get(key, False)

    # ------------------ save the results to a jsonl file ------------------
    score_output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(score_output_file, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logging.info(f"Saved score results to {score_output_file}")

    # ------------------ calculate the metrics and return ------------------ 
    metrics = _calculate_metrics(grouped_scores, items)
    final_eval_output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(final_eval_output_file, "w", encoding="utf-8") as f:
        for ds_name, ds_metrics in metrics.items():
            f.write(json.dumps([ds_name, ds_metrics], ensure_ascii=False) + "\n")
    logging.info(f"Saved final metrics to {final_eval_output_file}")

    if final_eval_csv_output_file is not None:
        final_eval_csv_output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(final_eval_csv_output_file, "w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "task",
                    "avg_k",
                    "pass_k",
                    "avg_total_tokens",
                    "avg_thinking_tokens",
                    "max_thinking_tokens",
                    "min_thinking_tokens",
                ],
            )
            writer.writeheader()
            for task_name, task_metrics in metrics.items():
                writer.writerow(
                    {
                        "task": task_name,
                        "avg_k": task_metrics.get("avg_k", 0),
                        "pass_k": task_metrics.get("pass_k", 0),
                        "avg_total_tokens": task_metrics.get("avg_total_tokens", 0),
                        "avg_thinking_tokens": task_metrics.get("avg_thinking_tokens", 0),
                        "max_thinking_tokens": task_metrics.get("max_thinking_tokens", 0),
                        "min_thinking_tokens": task_metrics.get("min_thinking_tokens", 0),
                    }
                )
        logging.info(f"Saved final metrics csv to {final_eval_csv_output_file}")
    
    return metrics