Installation
============

This guide covers how to install CortexFlow.

Prerequisites
------------

- Python 3.7 or higher
- Ollama (for local LLM integration)

Basic Installation
----------------

You can install CortexFlow using pip:

.. code-block:: bash

    pip install cortexflow

Installation with Optional Dependencies
------------------------------------

For graph-based knowledge features:

.. code-block:: bash

    pip install "cortexflow[graph]"

For the full package with all dependencies:

.. code-block:: bash

    pip install "cortexflow[all]"

Development Installation
---------------------

For development, you can install CortexFlow with development dependencies:

.. code-block:: bash

    git clone https://github.com/cortexflow/cortexflow.git
    cd cortexflow
    pip install -e ".[dev]"

Verifying Installation
-------------------

You can verify that CortexFlow is installed correctly by running:

.. code-block:: python

    import cortexflow
    print(cortexflow.__version__) 