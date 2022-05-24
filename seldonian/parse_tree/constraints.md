# Custom constraints
## Supervised learning case
It is possible that at some point you will want to enforce behavioral constraints that cannot be conveniently captured using the available interfaces. This library supports the addition of arbitrary custom user-defined base variables that you can include in your behavioral constraints. The only prerequisite is that you must be able to write a function, <img src="https://render.githubusercontent.com/render/math?math=\hat{z}\mathrm{(model,weights,data_dict)}">, that provides (ideally) unbiased estimates of that base variable. 

For example, consider the Seldonian regression algorithm in the [Thomas et al. 2019 Science paper (see Figure 2)](https://www.science.org/stoken/author-tokens/ST-119/full), designed to enforce fairness in GPA prediction between male and female applicants based on SAT test scores. The specific fairness constraint enforced in that paper was: the mean prediction error between male and female students should not differ by more 0.05 GPA points. This constraint is actually capturable in our interface via a single constraint using existing base variables: `abs((Mean_Error | [M]) - (Mean_Error | [F])) - epsilon`, where "Mean_Error" is a recognized base variable. 

However, it turns out that a tighter confidence bound can be obtained using a custom method for calculating the mean error difference. This alternative method pairs up the male and female observations and provides unbiased estimates of the mean error difference all in one base variable, which we will call `MED_MF`. This custom base variable is included in our library, called `MEDCustomBaseNode` in [constraints.py](./constraints.py). While following along with the instructions below, it may help to refer to the implementation of this example custom base node. 

To implement a custom base variable, follow these steps:

1. Define a class in [constraints.py](./constraints.py) that inherits from the `BaseNode` class, ideally conforming to [upper camel case](https://en.wikipedia.org/wiki/Camel_case) and named something that uniquely identifies your custom base variable. The name must not already be an existing class in that file.
2. Define an `__init__()` method for the class which at bare minimum takes a string expression as an argument. In the body of `__init__()` call the `__init__()` function of the parent class and pass it the `str_expression` argument, like so: `super().__init__(str_expression)`
3. Define a method for your class called `zhat`. Design this method to provide a vector of unbiased estimates of your base variable. 
5. At a minimum, the `zhat` method must take as input a model object, weight vector and data dictionary.
6. If your custom base variable requires preprocessing of the feature and/or label data before computing the bound, and the preprocessing is not dependent on the model weights, then define a method for your class called `calculate_data_forbound(**kwargs)` which returns a data dictionary and the number of observations in the potentially modified data, datasize. This will override this method from the parent class, `BaseNode`. The idea here is that there may be some preprocessing that you don't need to re-run each time you compute the bound in candidate selection. This method runs once at the beginning of candidate selection, and the result gets cached in the parse tree. Each subsequent time the confidence bound is calculated in candidate selection, the cached data is accessed rather than recomputed, potentially speeding up candidate selection enormously. 
7. Define a string expression for your custom base variable. It need not be a formula for the constraint, but it does need to uniquely identify this of constraint (or set of constraints). 
	- The string may only consist only of alphabet characeters (upper and lower case allowed) and the underscore character. No spaces are allowed. 
	- The string must not already be an existing key of the `custom_base_node_dict` dictionaries in [constraints.py](./constraints.py).
8. Add an entry to the dictionary: `custom_base_node_dict` in that file, where the key is the string expression of your constraint and the value is  is the name of the class you defined in step 2. 
9. In the interface (TODO: design interface to accept "custom-type" constraints) supply your custom behavioral constraints along with any parameters you wish to pass to your custom constraint class and the value of <img src="https://render.githubusercontent.com/render/math?math=\delta"> for the constraint. Note that you may supply normal constraints alongside custom constraints.  

These fairness constraints are actually capturable in our interfaces via a single constraint: `abs((Mean_Error | [M]) - (Mean_Error | [F])) - epsilon`, where `epsilon` is a constant. However, it turns out that the confidence bound obtained using the two custom constraints is tighter than the one written using the interface. The ghat methods pair up the male and female observations in a way that cannot be done with the normal interfaces.

As mentioned above in step 6, The `precalculate_data()` method is used to speed up candidate selection. In this example, before the male and female data are paired up in the ghat methods, the feature and label vectors need to be filtered so that they have the same number of female and male observations. Importantly, this filtering does not depend on theta, the current weight vector of the model, so it only needs to be done once in candidate selection (and once in the safety test). Both ghat methods are written so that they accept the same data after it has been modified in `precalculate_data()`. 

(TODO: provide screenshot and more specific instructions once we have designed the interface) To supply these custom constraints to the interface, for each custom constraint, we:
- Indicate that we are using a custom constraint.
- Provide the string expression for the constraint
- Provide the value of epsilon (tolerance) for each constraint, in this case 0.05.
- Provide the value of delta (confidence threshold) for each constraint, in this case 0.025.


