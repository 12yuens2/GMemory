from typing import Optional, Union
import logging
import string
import re

from langchain_core.documents import Document
import wikipedia

logger = logging.getLogger(__name__)

# Wikimedia refuses the `wikipedia` package's own default User-Agent, answering
# 403 with a text/plain body that the package then fails to parse as JSON. Its
# policy asks for a descriptive agent identifying the project:
# https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy
wikipedia.wikipedia.USER_AGENT = (
    "Intrinsic-Memory/0.1 "
    "(research; https://github.com/alan-turing-institute/Intrinsic-Memory)"
)
# The package still defaults to http, which Wikimedia redirects.
wikipedia.wikipedia.API_URL = "https://en.wikipedia.org/w/api.php"


class WikipediaUnavailable(RuntimeError):
    """Wikipedia could not be reached, as distinct from a page not existing.

    Kept separate because a run that cannot reach Wikipedia at all still scores:
    every Search returns nothing found, every episode fails, and the results look
    like a weak agent rather than a broken one.
    """


class LangChainWiki:

    def __init__(self) -> None:
        self.document: Optional[Document] = None
        self.lookup_str = ""
        self.lookup_index = 0

    def search(self, search: str) -> Union[str, Document]:
        try:
            page = wikipedia.page(search)
            result: Union[str, Document] = Document(
                page_content=page.content, metadata={"page": page.url}
            )
        except (wikipedia.PageError, wikipedia.DisambiguationError):
            result = f"Could not find [{search}]. Similar: {self._similar(search)}"
        except Exception as unreachable:
            logger.warning(
                "Wikipedia lookup of %r failed: %s: %s",
                search, type(unreachable).__name__, unreachable,
            )
            raise WikipediaUnavailable(
                f"Wikipedia could not be reached while searching for {search!r}: "
                f"{type(unreachable).__name__}: {unreachable}"
            ) from unreachable

        if isinstance(result, Document):
            self.document = result 
            return self._sumary
        else:
            self.document = None
            return result

    @staticmethod
    def _similar(search: str) -> list:
        """Titles close to `search`, or an empty list if even that fails."""
        try:
            return wikipedia.search(search)
        except Exception:
            return []
    
    def lookup(self, term: str):

        if self.document is None:
            raise ValueError("Cannot lookup without a successful search first")
        if term.lower() != self.lookup_str:
            self.lookup_str = term.lower() 
            self.lookup_index = 0
        else:
            self.lookup_index += 1
        lookups = [p for p in self._paragraphs if self.lookup_str in p.lower()]
        if len(lookups) == 0:
            return "No Results"
        elif self.lookup_index >= len(lookups):
            return "No More Results"
        else:
            result_prefix = f"(Result {self.lookup_index + 1}/{len(lookups)})"
            return f"{result_prefix} {lookups[self.lookup_index]}"

    @property
    def _sumary(self) -> str:
        return self._paragraphs[0]
    
    @property
    def _paragraphs(self) -> list[str]:
        if self.document is None:
            raise ValueError("Cannot get paragraphs without a document")
        return self.document.page_content.split("\n\n")
    


def normalize_answer(s: str):

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def match_exactly(answer, key) -> bool:

    n_answer = normalize_answer(answer)
    n_key = normalize_answer(key)
    return n_answer == n_key
