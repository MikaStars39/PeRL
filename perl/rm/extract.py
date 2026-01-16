async def call_qwen_extractor(text: str) -> str:
    """Call the background SGLang server for answer extraction."""
    extraction_prompt = (
        "You are a math answer extractor. Extract the final answer. "
        "Output ONLY the answer inside \\boxed{} if possible, or just the answer/number. "
        "Do not output explanation.\n\n"
        f"Text:\n{text}"
    )
    
    # SGLang is compatible with the OpenAI-style chat payload.
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": extraction_prompt}],
        "temperature": 0.0,
        "max_tokens": 128
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(CONFIG.sglang_url, json=payload) as resp:
                if resp.status == 200:
                    res = await resp.json()
                    return res['choices'][0]['message']['content'].strip()
                else:
                    logger.error(f"SGLang API Error: {resp.status}")
                    return ""
    except Exception as e:
        logger.error(f"SGLang Call Failed: {e}")
        return ""