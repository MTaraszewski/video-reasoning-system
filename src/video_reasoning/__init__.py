"""Find events in video from a plain-language description.

The public surface is one call:

    find_events(video_path, ["a forklift reverses"]) -> FindEventsResult

Everything else — frame sampling, the timestamp overlay, windowing, the model,
merging across window boundaries — is hidden behind it. That concealment is the
product: a client describes what it is looking for and gets back timecodes.
"""

__version__ = "0.1.0"
