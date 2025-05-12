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

Configuration options remain the same, just use the new class name:

.. code-block:: python

    # Old configuration
    config = AdaptiveContextConfig(
        active_token_limit=2000,
        working_token_limit=4000,
        archive_token_limit=6000
    )
    
    # New configuration
    config = CortexFlowConfig(
        active_token_limit=2000,
        working_token_limit=4000,
        archive_token_limit=6000
    )

Database Compatibility
-------------------

Knowledge stores created with AdaptiveContext are fully compatible with CortexFlow. No migration of data is necessary. 