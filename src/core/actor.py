from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.core.packet import Packet
from src.core.types import JsonDict
from src.providers.base import Provider


@dataclass(frozen=True)
class ActorResult:
    actor_id: str
    prompt_path: Path
    final_path: Path
    final_text: str
    events: list[JsonDict]
    metadata: JsonDict


@dataclass(frozen=True)
class Actor:
    """
    Provider-agnostic actor.

    - Builds a deterministic prompt from a file-based packet.
    - Calls provider.run(prompt).
    - Returns a normalized result (final_text + events + metadata).
    """

    actor_id: str
    packet: Packet
    provider: Provider
    include_paths_in_prompt: bool = True

    def build_prompt(self, *, extra_instructions: str | None = None) -> str:
        return self.packet.render(
            include_paths=self.include_paths_in_prompt,
            extra_instructions=extra_instructions,
        )

    def run(
        self,
        *,
        artifacts_dir: Path,
        timeout_s: float,
        idle_timeout_s: float,
        extra_instructions: str | None = None,
    ) -> ActorResult:
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        prompt = self.build_prompt(extra_instructions=extra_instructions)
        prompt_path = artifacts_dir / "prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        provider_result = self.provider.run(
            prompt,
            artifacts_dir=artifacts_dir,
            timeout_s=timeout_s,
            idle_timeout_s=idle_timeout_s,
        )

        final_text = provider_result.final_text
        final_path = artifacts_dir / "final.txt"
        final_path.write_text(final_text, encoding="utf-8")

        return ActorResult(
            actor_id=self.actor_id,
            prompt_path=prompt_path,
            final_path=final_path,
            final_text=final_text,
            events=provider_result.events,
            metadata=provider_result.metadata,
        )
