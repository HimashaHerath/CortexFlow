Installation
============

This guide covers how to install CortexFlow.

Prerequisites
------------

- Python 3.8 or higher
- Ollama (for local LLM integration) or Vertex AI credentials (for cloud-based inference)

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

For Vertex AI backend support:

.. code-block:: bash

    pip install "cortexflow[vertex]"

.. note::

   The ``vertex`` extra installs ``google-cloud-aiplatform`` and ``google-genai``
   for using Google Vertex AI models (e.g. Gemini) as an alternative to local
   Ollama inference. See the :doc:`configuration` guide for how to configure
   Vertex AI as your LLM backend.

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
