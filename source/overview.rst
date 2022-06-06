Overview
========

This overview will focus on the implementation of :term:`Seldonian algorithms<Seldonian Algorithm>` (SAs) in this library, rather than a description of what they are. For a detailed description of SAs see `the UMass AI Safety page <http://aisafety.cs.umass.edu/paper.html>`_, specifically `the Science paper <http://aisafety.cs.umass.edu/paper.html>`_. 

Often in a normal ML problem, data are split into training, testing (and sometimes validation) sets. SAs split data into :term:`candidate selection<Candidate Selection>` data and :term:`safety test<Safety test>` data. SAs consist of three main parts: the interface, candidate selection, and the safety test, which are described below. 

Interface
---------
In the interface, the user provides the data, metadata, the constraints that it wants the SA to enforce, and the confidence thresholds for each constraint. The interface creates a  :py:class:`.DataSet` object containing the data and metadata. The interface also creates :py:class:`.ParseTree` objects from the interpreted constraints. Finally, the interface generates a :py:class:`.Spec` object which consists of a complete specification that can used to run the seldonian algorithm in its entirety. The Spec object that the interface creates has many default parameters that the user may modify with their own custom script. 

Dataset
+++++++
The dataset object  See the :py:class:`.DataSet` objects API reference for a full description. 

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



