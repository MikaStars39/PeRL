from typing import Dict
from importlib import import_module

from .math.math_verify_reward import math_judge

# ----------------------- IMPORTANT: judge router -----------------------
# rule-based judge router that manage the judging process

def judge_router(
    response: str,
    label: str = "",
    source: str = None,
    **kwargs
) -> Dict:

    source_lower = (source or "").lower()
    if "ifeval" in source_lower:
        from .if_eval.if_eval import if_judge

        #
        # ifeval return:
        # return {
        #     'instruction_count': len(instructions),
        #     'instruction_pass_cnt': instruction_pass_cnt,
        #     'pass': prompt_level_pass_flag
        # }
        #
        return if_judge(response, **kwargs)
    elif "gpqa" in source_lower or "mmlu" in source_lower:
        from .gpqa.gpqa_verify_reward import gpqa_judge

        return gpqa_judge(response, label, **kwargs)
    elif any(key in source_lower for key in ("typos", "connections", "unscrambling")):
        language_module = import_module("nanoeval.reward.livebench.verify_language")
        language_judge = getattr(language_module, "language_judge")
        return language_judge(response, label, source, **kwargs)
    elif "tablejoin" in source_lower:
        table_module = import_module("nanoeval.reward.livebench.verify_table")
        table_process_results = getattr(table_module, "table_process_results")
        prompt = kwargs.get("prompt", "")
        return table_process_results(prompt, response, label)
    else:
        #
        # math return:
        # return {
        #     "pred": pred_ans,
        #     "pass": True if score == 1.0 else False
        # }
        #
        return math_judge(response, label, **kwargs)
