import warnings

warnings.warn(
    "lattereview.agents is deprecated and will be removed in v3.0. "
    "Use lattereview.agentic instead. Migration paths: "
    "BasicReviewer → lattereview.agentic.AgenticReviewer, "
    "ScoringReviewer → lattereview.agentic.ScoringReviewer, "
    "TitleAbstractReviewer → lattereview.agentic.TitleAbstractReviewer, "
    "AbstractionReviewer → lattereview.agentic.AbstractionReviewer.",
    DeprecationWarning,
    stacklevel=2,
)

from .basic_reviewer import BasicReviewer
from .scoring_reviewer import ScoringReviewer
from .abstraction_reviewer import AbstractionReviewer
from .title_abstract_reviewer import TitleAbstractReviewer
