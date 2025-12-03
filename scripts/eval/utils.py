import re

def extract_boxed_answer(response: str) -> str:
    """
    使用正则表达式提取 \\boxed{} 框住的答案，如果有多个则返回最后一个。
    
    任务：从响应文本中提取 LaTeX \\boxed{} 命令中的内容
    实现方案：
    1. 使用正则表达式找到所有 \\boxed{ 的位置
    2. 对于每个位置，通过匹配大括号来提取完整内容（处理嵌套情况）
    3. 返回最后一个匹配的结果，如果没有找到则返回空字符串
    """

    # 找到所有 \boxed{ 的位置
    pattern = r'\\boxed\{'
    matches = list(re.finditer(pattern, response))
    
    if not matches:
        return ""
    
    # 从最后一个匹配位置开始提取
    last_match = matches[-1]
    start_pos = last_match.end()  # \boxed{ 之后的位置
    
    # 手动匹配大括号，处理嵌套情况
    brace_count = 0
    content_start = start_pos
    content_end = start_pos
    
    for i in range(start_pos, len(response)):
        if response[i] == '{':
            brace_count += 1
        elif response[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                content_end = i
                break
    
    if brace_count == 0 and content_end > content_start:
        return response[content_start:content_end]
    
    return ""