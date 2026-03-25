"""UI support helpers shared across frontend-facing integration points."""

from .feedback import FeedbackRecord, append_feedback_record, feedback_summary

__all__ = [
    "FeedbackRecord",
    "append_feedback_record",
    "feedback_summary",
]
