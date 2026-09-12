"""eventfinder -- find described events in video with defensible timestamps.

    compile   turn the description into a Probe code can check
    signal    let a cheap change detector decide where to look at all
    observe   have the model describe what it sees, at times printed on frames
    derive    turn observations into timed, scored events in code

The model never localises and never holds state. It is asked what, never when.
"""
from .models import (
    SCHEMA_VERSION,
    Bracket,
    Event,
    EventsDocument,
    Observation,
    Probe,
    Rejection,
    RunInfo,
    Signals,
)

__all__ = [
    "SCHEMA_VERSION",
    "Bracket",
    "Event",
    "EventsDocument",
    "Observation",
    "Probe",
    "Rejection",
    "RunInfo",
    "Signals",
]
