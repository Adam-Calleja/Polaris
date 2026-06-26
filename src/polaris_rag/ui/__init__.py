"""UI support helpers shared across frontend-facing integration points.

This package groups the public helpers and types that belong to this subsystem of the
Polaris RAG codebase.

See Also
--------
feedback
    Related module for feedback.
"""

from .feedback import FeedbackRecord, append_feedback_record, feedback_summary

__all__ = [
    "FeedbackRecord",
    "append_feedback_record",
    "feedback_summary",
]
