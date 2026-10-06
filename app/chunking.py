"""Turn a markdown document into chunks that are good to embed.

Strategy (and why):
1. Split on markdown headings first. A heading marks a topic boundary, so chunks
   follow meaning instead of cutting at an arbitrary character count.
2. Prepend the heading path ("Partner Hospitals > Cancer care") to each chunk.
   A chunk saying only "bone marrow transplant, robotic surgery" is ambiguous;
   with its heading it carries context. This is called contextual chunk headers.
3. Only if a section is longer than CHUNK_MAX_CHARS, split it further on line /
   sentence boundaries, with a small overlap so facts at a boundary aren't lost.
"""
import re
from dataclasses import dataclass

HEADING_RE = re.compile(r"^(#{1,3})\s+(.*\S)\s*$")
SENTENCE_RE = re.compile(r"(?<=[.!?।])\s+")  # '।' is the Bangla full stop


@dataclass
class Chunk:
    text: str       # heading path + body: this is what gets embedded and shown to the LLM
    source: str     # file name
    heading: str    # e.g. "Partner Hospitals in India > Cancer care"
    lang: str       # "bn" or "en"
    index: int      # position within the file


def detect_lang(text: str) -> str:
    """Bangla if >30% of letters are in the Bengali Unicode block (U+0980-U+09FF)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    bn = sum(1 for c in letters if "\u0980" <= c <= "\u09ff")
    return "bn" if bn / len(letters) > 0.3 else "en"


def split_sections(markdown: str) -> list[tuple[str, str]]:
    """Return [(heading_path, body), ...]. A level-1 heading resets the path, which
    also cleanly separates the English and Bangla halves of a bilingual file."""
    stack: list[tuple[int, str]] = []
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in markdown.splitlines():
        m = HEADING_RE.match(line)
        if m:
            level, title = len(m.group(1)), m.group(2)
            stack = [(l, t) for l, t in stack if l < level] + [(level, title)]
            sections.append((" > ".join(t for _, t in stack), []))
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(body).strip()) for h, body in sections if "\n".join(body).strip()]


def _units(body: str, max_chars: int) -> list[str]:
    """Break a body into small units (lines; oversized lines into sentences)."""
    units: list[str] = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) <= max_chars:
            units.append(line)
        else:
            units.extend(s.strip() for s in SENTENCE_RE.split(line) if s.strip())
    return units


def split_long(body: str, max_chars: int, overlap: int) -> list[str]:
    """Greedily pack units into pieces <= max_chars; start each new piece with the
    tail of the previous one (up to `overlap` chars) for continuity."""
    if len(body) <= max_chars:
        return [body]
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for unit in _units(body, max_chars):
        if current and size + len(unit) + 1 > max_chars:
            pieces.append("\n".join(current))
            tail: list[str] = []
            tail_size = 0
            for u in reversed(current):
                if tail_size + len(u) > overlap:
                    break
                tail.insert(0, u)
                tail_size += len(u) + 1
            current, size = tail, tail_size
        current.append(unit)
        size += len(unit) + 1
    if current:
        pieces.append("\n".join(current))
    return pieces


def chunk_markdown(source: str, markdown: str, max_chars: int = 600, overlap: int = 100) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading, body in split_sections(markdown):
        for piece in split_long(body, max_chars, overlap):
            text = f"{heading}\n{piece}" if heading else piece
            chunks.append(Chunk(text=text, source=source, heading=heading,
                                lang=detect_lang(piece), index=len(chunks)))
    return chunks
