"""Typed errors.

Every failure says what was wrong, what was expected, and which parameter changes
it. Distinct types because the caller's response differs: "your file is not a video
I can read" and "you asked for something impossible" need different actions.

Exit codes are stable so scripts can branch on them.
"""
from __future__ import annotations


class VideoReasoningError(Exception):
    """Base for everything this package raises deliberately."""

    exit_code = 1

    def __init__(self, message: str, *, fix: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.fix = fix

    def __str__(self) -> str:
        return f"{self.message}\n  fix: {self.fix}" if self.fix else self.message


class InvalidInput(VideoReasoningError):
    """The request is malformed — a missing file, an empty query list, a value out
    of range. Nothing was attempted."""

    exit_code = 2


class UnprocessableMedia(VideoReasoningError):
    """The file exists but is not usable: not a video, no video stream, a duration
    outside supported bounds. Distinct from InvalidInput because the caller has to
    fix the *file*, not the request."""

    exit_code = 3


class Misconfigured(VideoReasoningError):
    """The configuration describes a system that cannot work — a stride wider than
    the window would leave footage nothing ever looks at."""

    exit_code = 4


class BudgetExceeded(VideoReasoningError):
    """The work implied by this request exceeds the configured budget. Raised
    before any model call, so nothing has been spent."""

    exit_code = 5


class BackendUnavailable(VideoReasoningError):
    """The model endpoint is unreachable, or is serving a different model than the
    one configured. The second case matters more: results attributed to the wrong
    model look like data rather than an error."""

    exit_code = 6
