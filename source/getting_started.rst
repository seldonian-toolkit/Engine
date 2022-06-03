Getting Started
===================

Let's run the linear regression example from the `AI Safety Tutorial <http://aisafety.cs.umass.edu/tutorial0.html>`_. Going through this example with acquaint us with many of the most important pieces of the libary. 

First, we will need to run an interface to tell the Seldonian algorithm what datasets and constraints we want to use. We will use the command line interface (CLI) in this example, specifically the supervised learning CLI.

Looking at the documentation for :py:mod:`.cli_supervised`, we can see the usage is:

 .. code-block:: console

     $ python interface.py data_pth metadata_pth
     [--include_sensitive_columns] 
     [--include_intercept_term]
     [--save_dir]

The two required arguments are :code:`data_pth`, the path to the data file and :code:`metadata_pth`, the path to the metadata file. The data and metadata files for this example can be downloaded from the source code repository on `GitHub <https://github.com/seldonian-framework/Engine/tree/main/static/datasets/GPA>`_. The other arguments are optional, but we will want to use 


