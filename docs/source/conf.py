# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# Project information
project = 'CortexFlow'
copyright = '2025, CortexFlow Team'
author = 'CortexFlow Team'

# Import project version
from cortexflow.version import __version__
release = __version__

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'myst_parser',
]

# Add mappings for intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# Templates
templates_path = ['_templates']
exclude_patterns = []

# Theme
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Logo
html_logo = '_static/logo.png'
html_favicon = '_static/favicon.ico'

# Custom sidebar templates
html_sidebars = {
    '**': [
        'globaltoc.html',
        'relations.html',
        'sourcelink.html',
        'searchbox.html',
    ]
}

# Napoleon settings for better docstring support
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False 