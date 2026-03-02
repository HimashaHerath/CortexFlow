Migration from AdaptiveContext
=============================

CortexFlow is the new name for the project previously known as AdaptiveContext. This guide helps existing users transition to the new package.

Simple Renaming
-------------

The simplest way to migrate is to update your imports:

.. code-block:: python

    # Old imports
    from adaptive_context import AdaptiveContextManager, AdaptiveContextConfig

    # New imports
    from cortexflow import CortexFlowManager, CortexFlowConfig

Class Name Changes
---------------

The following classes have been renamed:

+---------------------------+------------------------+
| Old Name                  | New Name               |
+===========================+========================+
| AdaptiveContextManager    | CortexFlowManager      |
+---------------------------+------------------------+
| AdaptiveContextConfig     | CortexFlowConfig       |
+---------------------------+------------------------+
| (module) adaptive_context | (module) cortexflow    |
+---------------------------+------------------------+

Temporary Compatibility Layer
--------------------------

For a transitional period, we maintain compatibility with old import paths:

.. code-block:: python

    # This still works but will show a deprecation warning
    from adaptive_context import AdaptiveContextManager, AdaptiveContextConfig

    # This is preferred
    from cortexflow import CortexFlowManager, CortexFlowConfig

The compatibility layer will be removed in a future version, so we recommend updating your code promptly.

Configuration Changes
------------------

CortexFlow now uses a nested configuration structure with dedicated sub-config
dataclasses instead of the flat parameter approach used in AdaptiveContext:

.. code-block:: python

    # Old flat configuration (AdaptiveContext)
    config = AdaptiveContextConfig(
        active_token_limit=2000,
        working_token_limit=4000,
        archive_token_limit=6000
    )

    # New nested configuration (CortexFlow)
    from cortexflow import CortexFlowConfig, MemoryConfig, LLMConfig

    config = CortexFlowConfig(
        memory=MemoryConfig(
            active_token_limit=2000,
            working_token_limit=4000,
            archive_token_limit=6000,
        ),
        llm=LLMConfig(default_model="llama3"),
    )

.. note::

   For backward compatibility, flat attribute access still works on
   ``CortexFlowConfig`` (e.g. ``config.active_token_limit``), but the nested
   style is recommended for all new code.

You can also use the ``ConfigBuilder`` for a fluent style:

.. code-block:: python

    from cortexflow import ConfigBuilder

    config = (ConfigBuilder()
        .with_memory(active_token_limit=2000, working_token_limit=4000, archive_token_limit=6000)
        .with_llm(default_model="llama3")
        .build())

Database Compatibility
-------------------

Knowledge stores created with AdaptiveContext are fully compatible with CortexFlow. No migration of data is necessary.
