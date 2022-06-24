Getting Started
===================


.. _installation:

Installation
------------

To use the Seldonian Engine, first install it using pip:

.. code-block:: console

   (.venv) $ pip install seldonian-engine

Then in Python:

.. code::
    
    import seldonian



If you want to visualize the parse tree graphs, a system-wide installation of `Graphviz <https://graphviz.org/download/>`_ is required.

.. _simple_example:

A simple, complete example
--------------------------
Consider a simple supervised regression problem with two continous random variables X and Y. Let the goal be to predict label Y using the single feature X. To solve this problem we could use univariate linear regression with an objective function of the mean squared error (MSE). We can find the optimal solution to this problem by minimizing the objective function w.r.t. to the weights of the model, :math:`{\theta}`, which in this case are just the intercept and slope of the line.

Now let's suppose we want to add the following two constraints into the problem:

1. Ensure that the MSE is less than 2.0 with a probability of at least 0.9. 
2. Ensure that the MSE is *greater than* 1.25 with a probability of at least 0.9. 

This problem can now be fully formulated as a Seldonian machine learning problem:

  Minimize the MSE, subject to the constraints:

    :math:`g_{1} = \mathrm{Mean\_Squared\_Error} - 2.0`, and :math:`{\delta}_1=0.1`. 
    
    :math:`g_{2} = 1.25 - \mathrm{Mean\_Squared\_Error}`, and :math:`{\delta}_2=0.1`. 

While the first constraint is aligned with the primary objective function, the MSE, notice that the second is conflicting. 

To code up this example using the engine, we need to follow these steps:

1. Define the data - we will generate some synthetic data for X and Y in this case.
2. Define the metadata - in this case this consists of the column names for X and Y and the regime, which is "supervised".
3. Put the data and metadata together into a DataSet object.
4. Define the behavioral constraints (constraint strings and confidence levels), which we already did above.
5. Make the parse trees from these behavioral constraints.
6. Define the underlying machine learning model and primary objective. 
7. Define an initial solution function which takes the features and labels as inputs and outputs an initial weight vector to start candidate selection. In this case we will define a function :code:`initial_solution()` function which just returns a zero vector as the initial solution.
8. Decide what fraction of the data to split into candidate selection vs. the safety test.
9. Decide what method to use for computing the high confidence upper bound on each :math:`g_{i}`. We will use Student :math:`t`-statistic.
10. Create a spec object containing all of this information and some hyperparameters - we can ignore many of these in this example. For a full list of parameters and their defaults see the API docs for :py:class:`.SupervisedSpec`.
11. Run the Seldonian algorithm using the spec object. 

Let's write out the code to do this. Each step above is enumerated in comments in the code below:

.. code::

    import autograd.numpy as np   # Thinly-wrapped version of Numpy
    import pandas as pd
    from seldonian.models.model import LinearRegressionModel
    from seldonian.dataset import SupervisedDataSet
    from seldonian.parse_tree.parse_tree import ParseTree
    from seldonian.spec import SupervisedSpec
    from seldonian.seldonian_algorithm import seldonian_algorithm

    
    if __name__ == "__main__":
        np.random.seed(0)
        numPoints=1000

        # 1. Define the data
        def generate_data(numPoints,loc_X=0.0,
            loc_Y=0.0,sigma_X=1.0,sigma_Y=1.0):
            """ The function we will use to generate 
            synthetic data
            """
            # Sample x from a standard normal distribution
            X = np.random.normal(loc_X, sigma_X, numPoints) 
            # Set y to be x, plus noise from a standard normal distribution
            Y = X + np.random.normal(loc_Y, sigma_Y, numPoints) 
            return (X,Y)
        X,Y = generate_data(numPoints)

        # 2. Define the metadataa
        columns = columns=['feature1','label']
        
        # 3. Make a dataset object
        rows = np.hstack([np.expand_dims(X,axis=1),
            np.expand_dims(Y,axis=1)])
        df = pd.DataFrame(rows,columns=columns)

        dataset = SupervisedDataSet(df,
            meta_information=columns,
            label_column='label',
            include_intercept_term=True)
        
        """ include_intercept_term=True
        adds a column of ones in the 
        feature array for convenience 
        during matrix multiplication.
        """

        # 4. Define the behavioral constraints
        constraint_strs = ['1.25 - Mean_Squared_Error','Mean_Squared_Error - 2.0']
        deltas = [0.1,0.1] # confidence levels

        # 5. Make the parse trees from these behavioral constraints 

        parse_trees = []
        for ii in range(len(constraint_strs)):
            constraint_str = constraint_strs[ii]

            delta = deltas[ii]

            # Create parse tree object
            parse_tree = ParseTree(
                delta=delta,
                regime='supervised',
                sub_regime='regression',
                columns=columns)

            # Fill out tree
            parse_tree.create_from_ast(constraint_str)
            # assign deltas for each base node
            # use equal weighting for each unique base node
            parse_tree.assign_deltas(weight_method='equal')

            # Assign bounds needed on the base nodes
            parse_tree.assign_bounds_needed()
            
            parse_trees.append(parse_tree)

        # 6. Define the underlying machine learning model and primary objective 
        model_class = LinearRegressionModel

        primary_objective = model_class().sample_Mean_Squared_Error

        # 7. Define initial solution function
        def initial_solution(X,y):
            """ Initial solution will be [0,0] """
            return np.zeros(2)

        initial_solution_fn=initial_solution

        # 8. Decide what fraction of your data to split into
        # candidate selection vs. the safety test.
        frac_data_in_safety=0.6
    
        """ 9. Decide what method to use for computing the
        high confidence upper bound on each :math:`g_{i}`.""" 
        bound_method='ttest'

        """10. Create a spec object, using some 
        hidden defaults we won't worry about here"""
        spec = SupervisedSpec(
            dataset=dataset,
            model_class=model_class,
            frac_data_in_safety=frac_data_in_safety,
            primary_objective=primary_objective,
            initial_solution_fn=initial_solution_fn,
            parse_trees=parse_trees,
            bound_method=bound_method,
        )

        # 11. Run seldonian algorithm using the spec object
        passed_safety,candidate_solution = seldonian_algorithm(spec)
        print(passed_safety,candidate_solution)

Notice in the last few lines that :code:`seldonian_algorithm()` returns two values. :code:`passed_safety` is a boolean indicating whether the candidate solution found during candidate selection passed the safety test. If :code:`passed_safety==False`, then :code:`candidate_solution="NSF"`, i.e. "No Solution Found". If :code:`passed_safety==True` then the candidate solution is a numpy array of model weights. In this example, you should get :code:`passed_safety=True` and a candidate solution of something like: :code:`[0.16911355 0.1738146]`, which might differ slightly depending on your machine's random number generator.



