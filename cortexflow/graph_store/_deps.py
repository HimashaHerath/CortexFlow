"""
Centralized optional dependency imports for the graph_store package.

All optional dependency checks and imports are performed here so that
sibling modules can simply ``from ._deps import ...`` the flags and
loaded modules they need.
"""
from __future__ import annotations

import logging

from cortexflow.dependency_utils import import_optional_dependency

# ---------------------------------------------------------------------------
# NetworkX
# ---------------------------------------------------------------------------
nx_deps = import_optional_dependency(
    'networkx',
    warning_message="networkx not found. Knowledge graph functionality will be limited."
)
NETWORKX_ENABLED = nx_deps['NETWORKX_ENABLED']
nx = nx_deps['module'] if NETWORKX_ENABLED else None

# ---------------------------------------------------------------------------
# spaCy
# ---------------------------------------------------------------------------
spacy_deps = import_optional_dependency(
    'spacy',
    warning_message="spacy not found. Automatic entity extraction will be limited."
)
SPACY_ENABLED = spacy_deps['SPACY_ENABLED']
spacy = spacy_deps['module'] if SPACY_ENABLED else None

# ---------------------------------------------------------------------------
# Flair (advanced NER)
# ---------------------------------------------------------------------------
flair_deps = import_optional_dependency(
    'flair.data',
    import_name='flair',
    warning_message="flair not found. Advanced entity recognition will be limited.",
    classes=['Sentence']
)
FLAIR_ENABLED = flair_deps['FLAIR_ENABLED']
Sentence = flair_deps.get('Sentence')
SequenceTagger = None

if FLAIR_ENABLED:
    try:
        from flair.models import SequenceTagger as _SequenceTagger
        SequenceTagger = _SequenceTagger
    except ImportError:
        FLAIR_ENABLED = False
        logging.warning("Flair SequenceTagger not available. Advanced NER is disabled.")

# ---------------------------------------------------------------------------
# SpanBERT (torch + transformers)
# ---------------------------------------------------------------------------
spanbert_deps = import_optional_dependency(
    'torch',
    warning_message="transformers/torch not found. SpanBERT entity recognition will be disabled."
)
transformers_deps = import_optional_dependency(
    'transformers',
    warning_message="",  # Skip duplicate warning
    classes=['AutoTokenizer', 'AutoModelForTokenClassification']
)
SPANBERT_ENABLED = spanbert_deps['TORCH_ENABLED'] and transformers_deps['TRANSFORMERS_ENABLED']
torch = spanbert_deps['module'] if SPANBERT_ENABLED else None
AutoTokenizer = transformers_deps.get('AutoTokenizer')
AutoModelForTokenClassification = transformers_deps.get('AutoModelForTokenClassification')

# ---------------------------------------------------------------------------
# Fuzzy matching (thefuzz)
# ---------------------------------------------------------------------------
fuzzy_deps = import_optional_dependency(
    'thefuzz',
    warning_message="thefuzz not found. Fuzzy entity matching will be disabled.",
    classes=['fuzz', 'process']
)
FUZZY_MATCHING_ENABLED = fuzzy_deps['THEFUZZ_ENABLED']
fuzz = fuzzy_deps.get('fuzz')
process = fuzzy_deps.get('process')

# ---------------------------------------------------------------------------
# Ontology (cortexflow internal)
# ---------------------------------------------------------------------------
ontology_deps = import_optional_dependency(
    'cortexflow.ontology',
    warning_message="Ontology module not found. Advanced knowledge graph capabilities will be limited.",
    classes=['Ontology']
)
ONTOLOGY_ENABLED = ontology_deps['CORTEXFLOW_ONTOLOGY_ENABLED']
Ontology = ontology_deps.get('Ontology')
