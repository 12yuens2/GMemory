def get_model_type(model_name: str) -> str:
# use this function for the model name so that something like "openai/gpt-oss-120b" will be recognized as "gpt-oss-120b" 
    valid_model_types: list[str] = [
        'gpt-4o-mini', 
        'qwen2.5-7b', 
        'qwen2.5-14b',
        'qwen2.5-32b', 
        'qwen2.5-72b',
        'intern', 
        'deepseek-v3',
        'llama3.2:1b',
        'mistral:7b',
        'llama3.2:3b',
        'qwen3:14b',
        'gpt-oss-120b',
    ]

    for model_type in valid_model_types:
        if model_type in model_name.lower():
            return model_type
    
    return 'unknown'
