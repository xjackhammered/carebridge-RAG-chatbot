"""Measure retrieval quality, and calibrate the 'refuse to answer' threshold.

    python -m eval.evaluate                                   # default model, default chunk size
    python -m eval.evaluate --models intfloat/multilingual-e5-base sentence-transformers/all-MiniLM-L6-v2 \
                            --chunk-sizes 300 600 1000

Metrics (only for the answerable questions):
  hit@k  - fraction of questions where a chunk containing the gold answer string
           (and from the right file) appears in the top k results.
  MRR    - mean reciprocal rank: 1/rank of the first correct chunk (0 if absent).
           1.0 means the right chunk is always first.

For the out-of-scope questions we look at the top-1 similarity score. If those
scores are clearly lower than the scores of answerable questions, a threshold
(MIN_SCORE) can separate them, and we search for the best one.
"""
import argparse
import json
import tempfile
from pathlib import Path

import chromadb

from app import config
from app.embedding import Embedder
from app.ingest import get_collection, ingest
from app.retriever import Retriever

QUESTIONS = Path(__file__).parent / "questions.jsonl"


def load_questions() -> list[dict]:
    return [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_hit(hit, q: dict) -> bool:
    """A chunk is correct if it contains ANY accepted gold answer: the primary one, or an
    alternative listed under "alt" (used when the corpus genuinely answers a question in two places)."""
    golds = [(q["source"], q["must_contain"])] + [tuple(a) for a in q.get("alt", [])]
    return any(hit.source == src and text in hit.text for src, text in golds)


def best_threshold(pos: list[float], neg: list[float]) -> tuple[float, float]:
    """Pick the cutoff that correctly classifies the most questions
    (answerable score >= t is answered, out-of-scope score < t is refused)."""
    best_t, best_acc = 0.0, 0.0
    for t in sorted(set(pos + neg)):
        correct = sum(s >= t for s in pos) + sum(s < t for s in neg)
        acc = correct / (len(pos) + len(neg))
        if acc >= best_acc:  # on ties prefer the higher (stricter) threshold
            best_t, best_acc = t, acc
    return best_t, best_acc


def run(embedder: Embedder, client, max_chars: int, k: int, questions: list[dict]) -> dict:
    ingest(client, embedder, config.DOCS_DIR, max_chars, config.CHUNK_OVERLAP)
    retriever = Retriever(get_collection(client, embedder.model_name, max_chars), embedder)

    rows, pos_scores, neg_scores = [], [], []
    for q in questions:
        hits = retriever.search(q["question"], k)
        if q.get("expect_no_answer"):
            neg_scores.append(hits[0].score)
            continue
        rank = next((i for i, h in enumerate(hits, 1) if is_hit(h, q)), None)
        pos_scores.append(hits[0].score)
        rows.append({"id": q["id"], "lang": q["lang"], "rank": rank, "question": q["question"],
                     "gold": q["source"], "top1_source": hits[0].source, "top1_text": hits[0].text,
                     "top1_score": hits[0].score})

    def metrics(subset):
        n = len(subset) or 1
        return {
            "n": len(subset),
            "hit@1": sum(1 for r in subset if r["rank"] == 1) / n,
            "hit@3": sum(1 for r in subset if r["rank"] and r["rank"] <= 3) / n,
            f"hit@{k}": sum(1 for r in subset if r["rank"]) / n,
            "mrr": sum(1 / r["rank"] for r in subset if r["rank"]) / n,
        }

    t, acc = best_threshold(pos_scores, neg_scores)
    return {
        "all": metrics(rows),
        "en": metrics([r for r in rows if r["lang"] == "en"]),
        "bn": metrics([r for r in rows if r["lang"] == "bn"]),
        "misses": [r for r in rows if not r["rank"]],
        "not_first": [r for r in rows if r["rank"] != 1],
        "pos_top1": (min(pos_scores), sum(pos_scores) / len(pos_scores)),
        "neg_top1": (max(neg_scores), sum(neg_scores) / len(neg_scores)) if neg_scores else None,
        "threshold": t,
        "threshold_acc": acc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[config.EMBEDDING_MODEL])
    ap.add_argument("--chunk-sizes", nargs="+", type=int, default=[config.CHUNK_MAX_CHARS])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--show-misses", action="store_true",
                    help="print every question whose correct chunk was not ranked first, with the chunk that won")
    args = ap.parse_args()

    questions = load_questions()
    lines = [f"| Model | Chunk | Lang | n | hit@1 | hit@3 | hit@{args.k} | MRR |", "|---|---|---|---|---|---|---|---|"]
    notes = []

    with tempfile.TemporaryDirectory() as tmp:  # throwaway DB so eval never touches your real index
        client = chromadb.PersistentClient(path=tmp)
        for model in args.models:
            embedder = Embedder(model)
            for size in args.chunk_sizes:
                res = run(embedder, client, size, args.k, questions)
                for lang in ("all", "en", "bn"):
                    m = res[lang]
                    lines.append(f"| {model.split('/')[-1]} | {size} | {lang} | {m['n']} | {m['hit@1']:.2f} | "
                                 f"{m['hit@3']:.2f} | {m[f'hit@{args.k}']:.2f} | {m['mrr']:.2f} |")
                if args.show_misses:
                    print(f"\n=== {model} / chunk {size}: questions not answered at rank 1 ===")
                    for r in res["not_first"]:
                        snippet = r["top1_text"].replace("\n", " ")[:160]
                        print(f"{r['id']} rank={r['rank']} top1={r['top1_score']:.3f} want={r['gold']} got={r['top1_source']}\n"
                              f"   Q: {r['question']}\n   TOP1: {snippet}")
                neg = res["neg_top1"]
                notes.append(
                    f"- **{model.split('/')[-1]} / chunk {size}**: answerable top-1 score min/avg = "
                    f"{res['pos_top1'][0]:.3f}/{res['pos_top1'][1]:.3f}; out-of-scope top-1 max/avg = "
                    + (f"{neg[0]:.3f}/{neg[1]:.3f}" if neg else "n/a")
                    + f"; best MIN_SCORE = **{res['threshold']:.3f}** ({res['threshold_acc']:.0%} of questions classified correctly)."
                    + (f" Missed: {', '.join(r['id'] for r in res['misses'])}." if res["misses"] else ""))

    report = "\n".join(lines) + "\n\n" + "\n".join(notes) + "\n"
    print(report)
    (Path(__file__).parent / "results.md").write_text(report, encoding="utf-8")
    print("Saved to eval/results.md")


if __name__ == "__main__":
    main()