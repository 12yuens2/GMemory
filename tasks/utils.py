import re


def model_dir_name(model_name: str) -> str:
    """A model name as a single path component: 'openai/gpt-oss-120b' -> 'openai--gpt-oss-120b'.

    Every model name gets a distinct directory, so two models nobody anticipated
    cannot write their results into the same files.
    """
    flattened = model_name.replace('/', '--')
    return re.sub(r'[^A-Za-z0-9._-]+', '-', flattened).strip('.-') or 'unnamed-model'
