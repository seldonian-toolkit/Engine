Overview
========

This overview will focus on the implementation of :term:`Seldonian algorithms<Seldonian Algorithm>` (SAs) in this library, rather than a description of what they are. For a detailed description of SAs see `the UMass AI Safety page <http://aisafety.cs.umass.edu/paper.html>`_, specifically `the Science paper <http://aisafety.cs.umass.edu/paper.html>`_. 

A SA is implemented by enforcing high probability constraints in a machine learning (ML) problem. This paradigm is extremely general, but often these constraints are safety or fairness constraints. Below, terms that have special significance in SAs and throughout this library have hyperlinks to the  glossary upon their first appearance. 

Example
-------
As a concrete example, consider a supervised ML problem designed to predict student grade point averages (GPAs) given a set of features about the students, such as standardized test scores. 

Let's say we want to use linear regression with a mean squared error loss function as our model to predict the GPAs, a float between 0.0 and 4.0. It is possible that this model will be biased against a particular gender class, *even if we did not use gender as a feature in the model*. For example, it could over predict GPAs for men and under predict GPAs for women. A constraint we might want to enforce to mitigate this bias is that the absolute difference in the prediction error between men and women may not exceed some constant, e.g. 0.2 GPA points. 

To interpret this example problem in the language of this library, the :term:`regime<Regime>` is supervised learning, the :term:`sub-regime<Sub-regime>` is regression, linear regression is the :term:`machine learning model<Machine learning model>`, mean squared error is the :term:`primary objective function<Primary objective function>`, gender is the :term:`sensitive attribute<Sensitive attribute>`. In general, there can be multiple sensitive attributes. It is the job of the user of the SA to provide the :term:`constraint<Behavioral Constraint>`. We provide several :term:`interfaces<Interface>` (see below) for interpreting constraints mathematically so that they can be used in the library. It is not possible to satisfy the constraint 100\% of the time. Instead, we must provide some :term:`confidence threshold<Confidence threshold>`, delta, such that the constraint will be enforced with probability :code:`1-delta`. 

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



Candidate Selection
-------------------
Explain candidate selection



