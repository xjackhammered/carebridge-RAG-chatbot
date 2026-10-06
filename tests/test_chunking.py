import json
from pathlib import Path

import pytest

from app.chunking import chunk_markdown, detect_lang, split_sections

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data" / "docs"


def test_detect_lang():
    assert detect_lang("How many hospitals?") == "en"
    assert detect_lang("ভারতে কতগুলো হাসপাতাল আছে?") == "bn"
    assert detect_lang("Dr. Rahim 09600-555010") == "en"


def test_heading_path_and_bilingual_split():
    md = "# Title\n\n## Sub\nBody one.\n\n# শিরোনাম\n\n## উপ\nবাংলা লেখা।"
    sections = dict(split_sections(md))
    assert "Title > Sub" in sections
    assert "শিরোনাম > উপ" in sections  # level-1 heading reset the path


def test_long_section_is_split_with_size_limit():
    body = "\n".join(f"- Item number {i} has some descriptive text attached." for i in range(40))
    chunks = chunk_markdown("x.md", f"# T\n\n## S\n{body}", max_chars=300, overlap=60)
    assert len(chunks) > 3
    assert all(len(c.text) <= 300 + len("T > S\n") for c in chunks)
    assert all(c.text.startswith("T > S") for c in chunks)  # every chunk keeps its heading


def test_overlap_repeats_boundary_line():
    body = "\n".join(f"Line {i:02d} " + "x" * 40 for i in range(10))
    chunks = chunk_markdown("x.md", f"# T\n{body}", max_chars=150, overlap=60)
    first_last = chunks[0].text.splitlines()[-1]
    assert first_last in chunks[1].text


@pytest.mark.parametrize("size", [300, 600, 1000])
def test_every_gold_answer_survives_chunking(size):
    """If chunking cut a gold answer string in half, eval would report a false miss."""
    questions = [json.loads(l) for l in (ROOT / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    for q in questions:
        if not q.get("source"):
            continue
        chunks = chunk_markdown(q["source"], (DOCS / q["source"]).read_text(encoding="utf-8"), size, 100)
        assert any(q["must_contain"] in c.text for c in chunks), f"{q['id']} lost at chunk size {size}"
