"""trend_method_v2_1 package.

Skeleton package for structured feature extraction, composite scoring,
and rule-based decisioning for epoch_tournament.
"""

from .config_v21 import get_v21_config
from .feature_builders_v21 import add_method_v21_features, apply_trend_method_v2_1
from .rule_decider_v21 import trend_method_v2_1_rules

__all__ = [
    "get_v21_config",
    "add_method_v21_features",
    "apply_trend_method_v2_1",
    "trend_method_v2_1_rules",
]
