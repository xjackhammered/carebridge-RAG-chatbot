"""Prompt text and small language helpers.

Notice what is NOT in the system prompt any more: hospital counts, doctor lists,
specialties. In your old project those facts were hardcoded in the prompt, so they
went stale and bypassed retrieval. Now every fact must come from the retrieved
context, which makes answers traceable to a source file.
"""
from app.chunking import detect_lang
from app.retriever import Hit

COMPANY_PHONE = "09600-555010"

NO_ANSWER = {
    "en": f"I don't have that information. Please contact CareBridge directly at {COMPANY_PHONE}.",
    "bn": f"এই তথ্যটি আমার কাছে নেই। অনুগ্রহ করে সরাসরি কেয়ারব্রিজে যোগাযোগ করুন {COMPANY_PHONE}।",
}

SYSTEM_PROMPT = f"""You are the assistant for CareBridge Health Services, a healthcare facilitation company in Dhaka, Bangladesh.

Rules:
1. Answer ONLY from the numbered context passages below. Never use outside knowledge and never invent doctors, hospitals, services, numbers or prices.
2. If the context does not contain the answer, reply exactly with the no-answer message you are given.
3. Never give or estimate treatment costs. Tell the user to call {COMPANY_PHONE} for a personalised estimate.
4. Never state doctor availability or appointment times. Appointments are booked by phone only: {COMPANY_PHONE}.
5. Reply in the same language as the user's question (Bangla script -> Bangla, English -> English). Use the formal "আপনি" in Bangla.
6. Be concise and warm. After each fact, cite the passage number in square brackets, like [1] or [2].
7. If the user describes a medical emergency, tell them to call {COMPANY_PHONE} immediately."""


def build_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] (source: {h.source})\n{h.text}" for i, h in enumerate(hits, 1))


def build_messages(question: str, hits: list[Hit], history: list[dict]) -> list[dict]:
    lang = detect_lang(question)
    system = (SYSTEM_PROMPT
              + f"\n\nNo-answer message: {NO_ANSWER[lang]}"
              + f"\n\nCONTEXT:\n{build_context(hits)}")
    return ([{"role": "system", "content": system}]
            + history
            + [{"role": "user", "content": question}])
