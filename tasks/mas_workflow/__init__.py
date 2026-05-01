from mas.mas import MetaMAS
from .autogen import AutoGen
from .macnet import MacNet
from .dylan import DyLAN
from .coder_reviewer import CoderReviewer
from .single_agent import SingleAgent
from .autogen.autogen_mas import AutoGen as AutoGenMAS

MAS = {
    'autogen': AutoGen,
    'macnet': MacNet,
    'dylan': DyLAN,
    'coder_reviewer': CoderReviewer,
    'single_agent': SingleAgent,
    'autogen_mas': AutoGenMAS
}

def get_mas(mas_type: str) -> MetaMAS:

    if MAS.get(mas_type) is None:
        raise ValueError('Unsupported mas type.')
    return MAS.get(mas_type)() 