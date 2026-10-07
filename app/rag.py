"""The RAG pipeline in one place: retrieve -> gate -> generate."""
from dataclasses import dataclass

from app.chunking import detect_lang
from app.llm import LLM, LLMUnavailable
from app.prompts import NO_ANSWER, build_messages
from app.retriever import Hit, Retriever


def clean_answer(text: str) -> str:
    """LLMs sometimes emit look-alike Unicode: non-breaking hyphens (breaks phone numbers)
    and full-width brackets (breaks citation format). Normalise them."""
    for bad, good in {"\u2011": "-", "\u2010": "-", "【": "[", "】": "]"}.items():
        text = text.replace(bad, good)
    return text


@dataclass
class RagResult:
    answer: str
    sources: list[Hit]
    grounded: bool          
    model: str | None
    top_score: float | None


class RagPipeline:
    def __init__(self, retriever: Retriever, llm: LLM | None, top_k: int,
                 min_score: float, max_tokens: int):
        self.retriever, self.llm = retriever, llm
        self.top_k, self.min_score, self.max_tokens = top_k, min_score, max_tokens

    def answer(self, question: str, history: list[dict]) -> RagResult:
        hits = self.retriever.search(question, self.top_k)
        top = hits[0].score if hits else None
        lang = detect_lang(question)

        if not hits or top < self.min_score:
            return RagResult(NO_ANSWER[lang], [], False, "gate", top)

        if self.llm is None:
            raise LLMUnavailable("GROQ_API_KEY is not configured")

        messages = build_messages(question, hits, history)
        text, model = self.llm.complete(messages, self.max_tokens)
        text = clean_answer(text)

        if NO_ANSWER[lang] in text:
            return RagResult(text, [], False, model, top)
        return RagResult(text, hits, True, model, top)