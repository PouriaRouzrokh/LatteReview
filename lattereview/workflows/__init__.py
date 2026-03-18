import warnings

warnings.warn(
    "lattereview.workflows is deprecated and will be removed in v3.0. "
    "Use lattereview.agentic.AgenticWorkflow instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .review_workflow import ReviewWorkflow
