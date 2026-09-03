import os
from pathlib import Path
from typing import TYPE_CHECKING, Union, Any
import random
import json
from dataclasses import dataclass
import math

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# Resolved against this file rather than the working directory, so a job can be
# started from wherever the scheduler puts it.
REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_path(*parts: str) -> Path:
    """An absolute path to something in the repository."""
    return REPO_ROOT.joinpath(*parts)


def load_json(file_name: str) -> Union[list, dict]:

    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


def write_json(json_obj, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False, separators=(",", ": "))

def random_divide_list(lst: list[Any], k: int) -> list[list]:
    """
    Divides the list into chunks, each with maximum length k.

    Args:
        lst: The list to be divided.
        k: The maximum length of each chunk.

    Returns:
        A list of chunks.
    """
    if len(lst) == 0:
        return []
    
    random.shuffle(lst)
    if len(lst) <= k:
        return [lst]
    else:
        num_chunks = math.ceil(len(lst) / k)
        chunk_size = math.ceil(len(lst) / num_chunks)
        return [lst[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
    

_EMBEDDING_MODEL_CACHE = {} 

@dataclass
class EmbeddingFunc:
    """The local embedding model, loaded on first use rather than on construction.

    `device` matters on a GPU node: SentenceTransformer takes cuda:0 when CUDA is
    there, which is where the vLLM server under test lives. One is constructed
    per experiment, before the memory module is known, and seven of the twelve
    registered modules never embed anything - so loading it here would put torch
    and a model in every worker for nothing.
    """

    model_type: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"

    @property
    def func(self) -> "SentenceTransformer":
        # Imported lazily: sentence_transformers pulls in torch.
        from sentence_transformers import SentenceTransformer

        key = (self.model_type, self.device)
        if key not in _EMBEDDING_MODEL_CACHE:
            _EMBEDDING_MODEL_CACHE[key] = SentenceTransformer(self.model_type, device=self.device)

        return _EMBEDDING_MODEL_CACHE[key]

    def embed_documents(self, texts: list[str]) -> list[list]:
        return [self.func.encode(text).tolist() for text in texts]

    def embed_query(self, query: str) -> list:
        return self.func.encode(query).tolist()


