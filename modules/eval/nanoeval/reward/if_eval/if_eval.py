from .instructions_registry import INSTRUCTION_DICT

def if_judge(
    response: str,
    **kwargs
):
    if response is None:
        response = ""
    instructions = kwargs['instruction_id_list']
    kwargs_list = kwargs['kwargs']

    prompt_level_pass_flag = True
    instruction_pass_cnt = 0
    
    for instruction_id, kwargs in zip(instructions, kwargs_list):
        instruct = INSTRUCTION_DICT[instruction_id](instruction_id)
        
        supported_keys = instruct.get_instruction_args_keys()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_keys}
        
        instruct.build_description(**filtered_kwargs)
        passed = instruct.check_following(response)
        
        if passed:
            instruction_pass_cnt += 1
        else:
            prompt_level_pass_flag = False
    
    return {
        'instruction_count': len(instructions),
        'instruction_pass_cnt': instruction_pass_cnt,
        'pass': prompt_level_pass_flag
    }