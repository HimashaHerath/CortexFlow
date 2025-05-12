"""
CortexFlow: Multi-tier memory optimization for LLMs with cognitive-inspired architecture.

This package provides tools to optimize context windows by implementing a
cognitive-inspired multi-tier memory system with dynamic allocation.
"""

from cortexflow.manager import CortexFlowManager
from cortexflow.config import CortexFlowConfig
from cortexflow.memory import ConversationMemory
from cortexflow.version import __version__

__all__ = [
    "CortexFlowManager",
    "CortexFlowConfig",
    "ConversationMemory",
    "__version__",
]

# Legacy support for adaptive_context
# This allows existing code to continue working with the new package
import warnings

# Issue a deprecation warning when importing via old names
def _warn_deprecated():
    warnings.warn(
        "The 'adaptive_context' package has been renamed to 'cortexflow'. "
        "Please update your imports. Support for old import paths will be "
        "removed in a future version.",
        DeprecationWarning,
        stacklevel=3
    )

# Legacy class names for backward compatibility
class AdaptiveContextManager(CortexFlowManager):
    def __init__(self, *args, **kwargs):
        _warn_deprecated()
        super().__init__(*args, **kwargs)

class AdaptiveContextConfig(CortexFlowConfig):
    def __init__(self, *args, **kwargs):
        _warn_deprecated()
        super().__init__(*args, **kwargs) 