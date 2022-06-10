Overview
========

This document provides an overview of how :term:`Seldonian algorithms<Seldonian Algorithm>` (SAs) are implemented using this library. For a detailed description of what SAs are, see `the UMass AI Safety page <http://aisafety.cs.umass.edu/overview.html>`_, specifically `the Science paper <http://aisafety.cs.umass.edu/paper.html>`_. 

At the broadest scope, SAs consist of three parts: the interface, candidate selection, and the safety test. Below are the main components of the API that you will interact within each of these.  

Interface
---------
The interface is used to provide:

- The `Data file`_ 
- The `Metadata file`_ 
- The `Behavioral constraints`_ you want the SA to enforce
- The :term:`Confidence level, delta<Confidence level>` at which you want each constraint to be enforced

There is currently one option for the interface, :py:mod:`.command_line_interface` (CLI). A graphical user interface (GUI) is currently in development. 

The interface generates a :py:class:`.Spec` object which consists of a complete specification that can used to run the seldonian algorithm in its entirety. The Spec object that the interface creates has many default parameters that the user may modify with their own custom script. 

Data file 
+++++++++
The data that you provide to the interface must be comma-separated and have no header. For example, a supervised learning dataset might look like:

.. code:: 

	0,1,622.6,491.56,439.93,707.64,663.65,557.09,711.37,731.31,509.8,1.33333
	1,0,538.0,490.58,406.59,529.05,532.28,447.23,527.58,379.14,488.64,2.98333
	1,0,455.18,440.0,570.86,417.54,453.53,425.87,475.63,476.11,407.15,1.97333
	0,1,756.91,679.62,531.28,583.63,534.42,521.4,592.41,783.76,588.26,2.53333
	...

This file should include *all* of the data you have, i.e. the data you have before splitting into train,test,validation splits. The Seldonian algorithm with partition your data for you. The column names are intentionally excluded from this file and are provided in the `Metadata file`_. 

Metadata file 
+++++++++++++
The metadata file is a JSON-formatted file containing important properties of your dataset. It has different required keys depending on the :term:`Regime` of your problem. For supervised learning, the required keys are:

- "regime", which is set to 'supervised' in this case
- "sub_regime", which is either 'classification' or 'regression'
- "columns", a list of the column names in your `Data file`_. 
- "label_column", the column that you are trying to predict
- "sensitive_columns", a list of the column names of the :term:`sensitive attributes <Sensitive attribute>` 

For reinforcement learning, the required keys are:

- "regime", which is set to 'RL' in this case
- "columns", a list of the column names in your `Data file`_. 
- "RL_environment_name", the name of the module in :py:mod:`.RL.environments` package containing the RL Environment() class you want to use. 

Behavioral constraints
++++++++++++++++++++++
In the `definition of a Seldonian algorithm <http://aisafety.cs.umass.edu/tutorial1.html>`_, :term:`behavioral constraints<Behavioral constraint>`, :math:`(g_i,{\delta}_i)_{i=1}^n` are of a set of constraint functions, :math:`g_i`, and confidence levels, :math:`{\delta}_i`. Constraint functions are not provided to the interface directly, but are built by the engine from *constraint strings* provided by the user. 

Constraint strings must contain the mathematical definition of the constraints, :math:`g_i`. Certain statistical functions (called "measure functions") have special strings associated with them so that they the engine recognizes them when they appear in the constraint string. For a list of these measure functions, see: :py:mod:`.parse_tree.operators`. 

Examples of valid basic *constraint strings* and their plain English definitions are below. If you find these definitions confusing, remember that we want :math:`g_i<=0` to be satisfied.

- :code:`Mean_Squared_Error - 2.0`: "Ensure that the mean squared error is less than 2.0". Here, :code:`Mean_Squared_Error` is a special measure function. 

- :code:`0.88 - TPR`: "Ensure that the True Positive Rate is greater than 0.88".

These basic constraint strings cover a number of use cases, but they apply to the whole dataset. Often one desires a constraint that apply to a sensitive attribute of the dataset. Some examples for this use case include:

- :code:`abs((PR | [M]) - (PR | [F])) - 0.15`: "Ensure that the absolute difference between the positive rate for males (M) and the positive rate for females (F) is less than 0.15". This constraint is called demographic parity (with a tolerance of 15%). 





Parse Trees
+++++++++++
Explain parse trees.

Specification (Spec) object
+++++++++++++++++++++++++++
Explain spec object


.. _candidate_selection:

Candidate Selection
-------------------

Explain candidate selection
Explain hyperparams



