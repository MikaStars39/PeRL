import re

def extract_answer(text: str) -> str:
    """Extract answer from model response using regex (boxed or last value)."""
    if not text:
        return ""

    # NOTE:
    # - 一些 jsonl 里可能错误地写成 "\boxed{...}"（单反斜杠）。
    #   json.loads 会把 "\b" 解析成退格符 \x08，导致后续正则匹配不到。
    #   这里把退格符还原为字面量 "\b"（两字符：反斜杠 + b）。
    if "\x08" in text:
        text = text.replace("\x08", "\\b")

    # 1) 优先提取 \boxed{...}
    # 不能用简单正则去找 "第一个 }" 结束，因为 boxed 内容里常见嵌套花括号：
    #   \boxed{9.0 \times 10^{11}}
    # 这里用括号配对解析，确保提取完整 boxed 内容；若有多个，取最后一个。
    results = []
    for m in re.finditer(r"\\boxed\b", text):
        i = m.end()
        # 跳过 \boxed 后面的空白，找到第一个 '{'
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text) or text[i] != "{":
            continue

        i += 1  # skip '{'
        depth = 1
        start = i
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            i += 1

        if depth == 0:
            # i 已经指向匹配到的 '}' 之后
            results.append(text[start : i - 1].strip())

    if results:
        return results[-1]
    else:
        return None

def gpqa_judge(
    response: str,
    label: str = "",
    **kwargs
) -> dict:
    raw_eval_res = response
    pred_ans = extract_answer(raw_eval_res)
    
    if not pred_ans:
        return {
            "pred": pred_ans,
            "pass": False
        }
    
    if pred_ans == label:
        return {
            "pred": pred_ans,
            "pass": True
        }
    else:
        if label.lower() in pred_ans.lower():
            return {
                "pred": pred_ans,
                "pass": True
            }
        else:
            return {
                "pred": pred_ans,
                "pass": False
            }