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

The interface generates a `Spec object`_ object which consists of a complete specification that can used to run the seldonian algorithm in its entirety. The Spec object that the interface creates has many default parameters that the user may modify with their own custom script. 

Data file 
+++++++++
The data that you provide to the interface must be rows of numbers that are comma-separated and have no header. The rows are separated by newlines. For example, a supervised learning dataset might look like:

.. code:: 

	0,1,622.6,491.56,439.93,707.64,663.65,557.09,711.37,731.31,509.8,1.33333
	1,0,538.0,490.58,406.59,529.05,532.28,447.23,527.58,379.14,488.64,2.98333
	1,0,455.18,440.0,570.86,417.54,453.53,425.87,475.63,476.11,407.15,1.97333
	0,1,756.91,679.62,531.28,583.63,534.42,521.4,592.41,783.76,588.26,2.53333
	...

This file should include *all* of the data you have, i.e. the data you have before splitting into train,test,validation splits. The Seldonian algorithm will partition your data for you. The column names are intentionally excluded from this file and are provided in the `Metadata file`_. 

Metadata file 
+++++++++++++
The metadata file is a JSON-formatted file containing important properties about your dataset. It has different required keys depending on the :term:`Regime` of your problem. For supervised learning, the required keys are:

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

Constraint strings contain the mathematical definition of the constraint functions, :math:`g_i`. These strings are written as Python strings and support the following mathematical operators:

- :code:`+,-,*,/`

The following native Python mathematical functions are also supported: 

- :code:`min()`
- :code:`max()`
- :code:`abs()`
- :code:`exp()`

Certain statistical functions (called "measure functions") have special strings associated with them so that the engine recognizes them when they appear in the constraint string. For example, if :code:`Mean_Squared_Error` appears in the string it will be interpreted correctly by the engine. For a full list of these measure functions, see: :py:mod:`.parse_tree.operators`. 

Examples of the most basic constraint strings and their plain English definitions are below. Remember that in the Seldonian framework we want :math:`g_i{\leq}0` to be satisfied. The :math:`{\leq}0` is omitted from the constraint strings. 

- :code:`Mean_Squared_Error - 2.0`: "Ensure that the mean squared error is less than or equal to 2.0". Here, :code:`Mean_Squared_Error` is a special measure function for supervised regression problems. 

- :code:`0.88 - TPR`: "Ensure that the True Positive Rate (TPR) is greater than or equal to 0.88". Here, :code:`TPR` is a measure function for supervised classification problems.

- :code:`0.5 - J_pi_new`: "Ensure that the performance of the new policy (:code:`J_pi_new`) is greater than or equal to 0.5". Here, :code:`J_pi_new` is a measure function for RL problems.

These basic constraint strings cover a number of use cases. However, they do not use information about the sensitive attributes (columns) in the dataset, which commonly appear in fairness definitions. The generic specification for including sensitive attributes in the constraint string is as follows:

.. code::
	
	(measure_function | [ATR1,ATR2,...])

where :code:`measure_function` is a placeholder for the actual measure function in use and :code:`[ATR1,ATR2,...]` is a placeholder list of attributes (column names) from the dataset. The parentheses surrounding the statement are required in all cases.

The following examples show valid constraint strings that use sensitive attributes of an example dataset with sensitive attributes: :code:`[M,F,R1,R2]`. These only apply for the supervised learning regime. 

- :code:`abs((PR | [M]) - (PR | [F])) - 0.15`: "Ensure that the absolute difference between the positive rate for males (M) and the positive rate (PR, a measure function) for females (F) is less than or equal to 0.15". This constraint is called demographic parity (with a tolerance of 15%). Here, :code:`M` and :code`F` must be columns of the dataset, as specified in the `Metadata file`_. We also see the use of a native Python function, :code:`abs()` in this constraint string. 

- :code:`0.8 - min((PR | [M])/(PR | [F]),(PR | [F])/(PR | [M]))`: "Ensure that ratio of the positive rate for males (M) to the positive rate for females (F) or the inverse is at least 0.8." This constraint is called disparate impact (with a tolerance of 0.8). We see the use of :code:`min()`, yet another native Python function in this constraint string. 

It is permitted to use more than one attribute for a given measure function. For example:

- :code:`(FPR | [F,R1]) - 0.2`: "Ensure that the false positive rate (FPR) for females (F) belonging to race class 1 (RL) is less than or equal to 0.2. 

Note that the user must also specify the values of :math:`{\delta}` for each provided constraint string.


Spec object
+++++++++++
The :py:class:`.Spec` object (short for specification object) contains all of the inputs needed to run the Seldonian algorithm. The interface creates a spec object from the user's inputs. It is up to the designer of the interface to specify how the user will provide the information needed to complete the spec object. Because the spec object is editable, it may be practical to create a simple interface that generates a spec object with many default values and then require the user to modify the spec object in a custom script. 


.. _candidate_selection:

Candidate Selection
-------------------
Candidate selection is run inside of the :py:func:`.seldonian_algorithm.seldonian_algorithm` function. The inputs to candidate selection are assembled from the spec object provided to the function. First, a :py:class:`.CandidateSelection` object is created, then :py:meth:`.CandidateSelection.run` is called to start candidate selection. 

:code:`run()` returns the :code:`candidate_solution`, the optimized model weights obtained during candidate selection, or :code:`'NSF'` if no solution was found. 

There are currently two supported optimization techniques for candidate selection: 

1. Black box optimization with a barrier function. The barrier, which is shaped like the upper bound functions, is added to the cost function when any of the constraints are violated. This forces solutions toward the feasible set. 

2. Gradient descent on a `Lagrangian <https://en.wikipedia.org/wiki/Lagrange_multiplier#:~:text=In%20mathematical%20optimization%2C%20the%20method,chosen%20values%20of%20the%20variables).>`_:

.. math::

	{\mathcal{L(\mathbf{\theta,\lambda})}} = f(\mathbf{\theta}) + {\sum}_i^{n} {\lambda_i} g_i(\mathbf{\theta})

where :math:`\mathbf{\theta}` is the vector of model weights, :math:`f(\mathbf{\theta})` is the primary objective function, :math:`g_i(\mathbf{\theta})` is the ith constraint function of :math:`n` constraints, and :math:`\mathbf{\lambda}` is a vector of Lagrange multipliers, such that :math:`{\lambda_i}` is the Lagrange multiplier for the ith constraint. 

The `KKT <https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions>`_ Theorem states that the saddle points of :math:`{\mathcal{L}}` are optima of the constrainted optimization problem:

	Optimize :math:`f({\theta})` subject to:
		
		:math:`g_i({\theta}){\leq}0, {\quad} i{\in}\{0{\ldots}n\}`


To find the saddle points we use gradient descent to obtain the global minimum over :math:`{\theta}` and simultaneous gradient *ascent* to obtain the global maximum over the multipliers, :math:`{\lambda}`.

In situations where the contraints are conflicting with the primary objective, vanilla gradient descent can result in oscillations of the solution near the feasible set boundary. These oscillations can be dampened using momentum in gradient descent. We implemented the adam optimizer as part of our gradient descent method, which includes momentum, and found that it mitigates the oscillations in all problems we have tested so far. 

Safety Test
-----------
The safety test is run on the candidate solution returned by candidate selection. Like candidate selection, the safety test is run inside of the :py:func:`.seldonian_algorithm.seldonian_algorithm` function. The inputs to the safety test are assembled from the spec object provided to the function. First, a :py:class:`.SafetyTest` object is created, then :py:meth:`.SafetyTest.run` is called to start the safety test.  

