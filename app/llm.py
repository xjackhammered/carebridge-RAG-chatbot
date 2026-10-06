"""Groq client with an ordered model fallback list."""
import logging

from groq import Groq

log = logging.getLogger("llm")


class LLMUnavailable(Exception):
    pass


class LLM:
    def __init__(self, api_key: str, models: list[str]):
        self.client = Groq(api_key=api_key)
        self.models = models

    def available_models(self) -> set[str]:
        """Ask Groq which models exist right now. Used at startup to warn about
        deprecated models BEFORE a user hits an error."""
        return {m.id for m in self.client.models.list().data}

    def complete(self, messages: list[dict], max_tokens: int) -> tuple[str, str]:
        last_error = None
        for model in self.models:
            try:
                resp = self.client.chat.completions.create(
                    model=model, messages=messages, max_tokens=max_tokens, temperature=0.2)
                return resp.choices[0].message.content.strip(), model
            except Exception as e:  # rate limit, deprecated model, network...
                log.warning("model %s failed: %s", model, e)
                last_error = e
        raise LLMUnavailable(str(last_error))
