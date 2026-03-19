Development
===========

This section describes how to use ``msentity`` in development environments.

Using as a Git Submodule
------------------------

You can integrate ``msentity`` into an existing project as a Git submodule.

.. code-block:: bash

   # At the root directory of your project
   git submodule add https://github.com/MotohiroOGAWA/msentity.git ./msentity
   git commit -m "Add msentity as submodule"

Notes
-----

Using a Git submodule is useful for:

- Reproducible research environments
- Managing dependencies in HPC or cluster environments
- Integrating ``msentity`` into larger pipelines