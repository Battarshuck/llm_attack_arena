import os

model_names_list = {
    'phi2':"microsoft/phi-2",
    'llama':"meta-llama/Llama-3.2-1B",
    'deepseek': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
}

developers = {
    'phi2':"microsoft",
    'llama':"meta",
    'deepseek': 'deepseek-ai'
}

def get_model_path(model_name):
    return os.path.join(os.path.dirname(__file__), 'models', model_name)

def get_developer(model_name):
    if model_name in developers:
        return developers[model_name]
    else:
        raise ValueError(f"Unknown model name: {model_name}. Available models are {list(developers.keys())}.")
    