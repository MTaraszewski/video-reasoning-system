"""Prompts.

The brief says Cosmos 3 Edge "can localise events with timestamps **when prompted
correctly**". That qualifier is doing real work: prompt design is a first-class
experiment here, not a detail to settle afterwards. So prompts are named variants
that can be swept, and the probe reports which one won.

The `overlay` variant follows NVIDIA's published temporal-localisation recipe:
instruct the model to read timestamps printed on the frames, and answer in a
structured form. The `native` variant asks for time without mentioning overlays,
to test whether the model localises some other way.
"""
from __future__ import annotations

from dataclasses import dataclass

_JSON_CONTRACT = (
    'Respond with JSON only, in exactly this form:\n'
    '{"events": [{"start_s": <number>, "end_s": <number>, '
    '"confidence": <0-1>, "evidence": "<short phrase>"}]}\n'
    'If the event does not occur in this segment, respond {"events": []}.'
)


@dataclass(frozen=True)
class PromptVariant:
    name: str
    system: str
    user_template: str

    def user(self, query: str, start_s: float, end_s: float) -> str:
        return self.user_template.format(
            query=query, start_s=start_s, end_s=end_s, contract=_JSON_CONTRACT
        )


# Reads the burned-in overlay. NVIDIA documents this mechanism for Cosmos
# Reason 2; whether Cosmos3-Edge supports it is exactly what the probe tests.
OVERLAY = PromptVariant(
    name="overlay",
    system=(
        "You are a precise video event-localization system. You are shown frames "
        "sampled in temporal order from one segment of a longer video. Each frame "
        "has its absolute timestamp printed at the bottom-left, for example "
        "'t=12.375s'. Read those printed timestamps to report when things happen. "
        "Report only what is visible in the frames. Do not guess times you cannot "
        "read."
    ),
    user_template=(
        'Find every moment in this segment matching: "{query}".\n\n'
        "These frames span t={start_s:.3f}s to t={end_s:.3f}s. Every time you "
        "report must lie inside that range and must come from a timestamp you can "
        "actually read on a frame.\n\n"
        "{contract}"
    ),
)

# No mention of overlays: does the model place events in time some other way?
# If this wins, the burned-in timestamp is unnecessary work.
NATIVE = PromptVariant(
    name="native",
    system=(
        "You are a precise video event-localization system. You are shown a "
        "segment of video. Report when events occur, in seconds relative to the "
        "start of the whole video. Report only what is visible."
    ),
    user_template=(
        'Find every moment in this segment matching: "{query}".\n\n'
        "This segment covers t={start_s:.3f}s to t={end_s:.3f}s of the video. "
        "Report times within that range.\n\n"
        "{contract}"
    ),
)

# Minimal instruction — a control. If it matches the others, the elaborate
# prompting was not what made the difference.
TERSE = PromptVariant(
    name="terse",
    system="Answer with JSON only.",
    user_template=(
        'In these video frames (t={start_s:.3f}s to t={end_s:.3f}s), when does '
        'this happen: "{query}"?\n\n{contract}'
    ),
)

VARIANTS: dict[str, PromptVariant] = {v.name: v for v in (OVERLAY, NATIVE, TERSE)}
DEFAULT = OVERLAY


def get(name: str | None) -> PromptVariant:
    if not name:
        return DEFAULT
    if name not in VARIANTS:
        raise KeyError(f"unknown prompt variant {name!r}; have {sorted(VARIANTS)}")
    return VARIANTS[name]
