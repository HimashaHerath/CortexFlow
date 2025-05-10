"""
Registry for managing benchmark system adapters.
"""
from typing import Dict, Any, List, Type, Callable

# Import system adapters
try:
    from benchmark.adapters.adaptivecontext_adapter import AdaptiveContextAdapter
    ADAPTIVECONTEXT_AVAILABLE = True
except ImportError:
    ADAPTIVECONTEXT_AVAILABLE = False
    print("Warning: AdaptiveContext not available")

try:
    from benchmark.adapters.llamaindex_adapter import LlamaIndexAdapter, LLAMAINDEX_AVAILABLE
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    print("Warning: LlamaIndex not available")

try:
    from benchmark.adapters.langchain_adapter import LangChainAdapter, LANGCHAIN_AVAILABLE
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: LangChain not available")

try:
    from benchmark.adapters.activeloop_adapter import ActiveloopAdapter, ACTIVELOOP_AVAILABLE
except ImportError:
    ACTIVELOOP_AVAILABLE = False
    print("Warning: Activeloop DeepMemory not available")

try:
    from benchmark.adapters.qdrant_adapter import QdrantAdapter, QDRANT_AVAILABLE
except ImportError:
    QDRANT_AVAILABLE = False
    print("Warning: Qdrant not available")

# Map system names to their adapter classes
SYSTEM_ADAPTERS = {}

# Register available systems
if ADAPTIVECONTEXT_AVAILABLE:
    SYSTEM_ADAPTERS["adaptivecontext"] = AdaptiveContextAdapter

if LLAMAINDEX_AVAILABLE:
    SYSTEM_ADAPTERS["llamaindex"] = LlamaIndexAdapter

if LANGCHAIN_AVAILABLE:
    SYSTEM_ADAPTERS["langchain"] = LangChainAdapter

if ACTIVELOOP_AVAILABLE:
    SYSTEM_ADAPTERS["activeloop"] = ActiveloopAdapter

if QDRANT_AVAILABLE:
    SYSTEM_ADAPTERS["qdrant"] = QdrantAdapter


def get_available_systems() -> List[str]:
    """
    Get list of available benchmark systems.
    
    Returns:
        List of system names
    """
    return list(SYSTEM_ADAPTERS.keys())


def get_system_adapter(name: str) -> Type:
    """
    Get adapter class for a system.
    
    Args:
        name: System name
        
    Returns:
        Adapter class
        
    Raises:
        ValueError: If system not found or not available
    """
    name = name.lower()
    if name not in SYSTEM_ADAPTERS:
        available = ", ".join(get_available_systems())
        raise ValueError(f"System '{name}' not found. Available systems: {available}")
    
    return SYSTEM_ADAPTERS[name]


def register_system(name: str, adapter_class: Type) -> None:
    """
    Register a new system adapter.
    
    Args:
        name: System name
        adapter_class: Adapter class
    """
    SYSTEM_ADAPTERS[name.lower()] = adapter_class 