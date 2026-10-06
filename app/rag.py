"""The RAG pipeline in one place: retrieve -> gate -> generate."""
from dataclasses import dataclass

from app.chunking import detect_lang
from app.llm import LLM, LLMUnavailable
from app.prompts import NO_ANSWER, build_messages
from app.retriever import Hit, Retriever


@dataclass
class RagResult:
    answer: str
    sources: list[Hit]
    grounded: bool          # False -> we refused / could not answer from the documents
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

        # GATE: if even the best chunk is weakly related, skip the LLM entirely.
        # This is the main anti-hallucination mechanism: the model can't make
        # something up about a topic it was never given context for. It also
        # saves an API call.
        if not hits or top < self.min_score:
            return RagResult(NO_ANSWER[detect_lang(question)], [], False, "gate", top)

        if self.llm is None:
            raise LLMUnavailable("GROQ_API_KEY is not configured")

        messages = build_messages(question, hits, history)
        text, model = self.llm.complete(messages, self.max_tokens)
        return RagResult(text, hits, True, model, top)
