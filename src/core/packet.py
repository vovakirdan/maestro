from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DOC_ORDER: tuple[str, ...] = (
    "ROLE",
    "TARGET",
    "RULES",
    "CONTEXT",
    "REPORT_FORMAT",
    "INPUTS",
    "NOTES",
)


@dataclass(frozen=True)
class Document:
    name: str
    path: Path
    text: str


@dataclass(frozen=True)
class Packet:
    packet_dir: Path
    documents: dict[str, Document]

    @classmethod
    def load(cls, packet_dir: Path) -> Packet:
        docs: dict[str, Document] = {}
        for name in DOC_ORDER:
            path = packet_dir / f"{name}.md"
            try:
                text = path.read_text(encoding="utf-8")
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Missing packet document: {path}") from e
            docs[name] = Document(name=name, path=path, text=text)
        return cls(packet_dir=packet_dir, documents=docs)

    def render(self, *, include_paths: bool, extra_instructions: str | None = None) -> str:
        parts: list[str] = []
        for name in DOC_ORDER:
            doc = self.documents[name]
            parts.append(f"=== {name} ===")
            if include_paths:
                parts.append(f"SOURCE: {doc.path.as_posix()}")
            parts.append(doc.text.rstrip())
            parts.append("")

        if extra_instructions and extra_instructions.strip():
            parts.append("=== EXTRA_INSTRUCTIONS ===")
            parts.append(extra_instructions.rstrip())
            parts.append("")

        return "\n".join(parts).rstrip() + "\n"


def write_packet_documents(packet_dir: Path, *, docs: dict[str, str]) -> None:
    packet_dir.mkdir(parents=True, exist_ok=True)
    for name in DOC_ORDER:
        path = packet_dir / f"{name}.md"
        content = docs.get(name, "")
        if not content.endswith("\n"):
            content += "\n"
        path.write_text(content, encoding="utf-8")
