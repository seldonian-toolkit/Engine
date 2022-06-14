""" Module for running safety test """

import autograd.numpy as np   # Thinly-wrapped version of Numpy

class SafetyTest(object):
	def __init__(self,dataset,model,parse_trees,regime='supervised',**kwargs):
		""" 
		Object for running safety test
		
		:param dataset: The dataset object containing safety data
		:type dataset: :py:class:`.DataSet` object

		:param model: The Seldonian model object 
		:type model: :py:class:`.SeldonianModel` object

		:param parse_trees: List of parse tree objects containing the 
			behavioral constraints
		:type parse_trees: List(:py:class:`.ParseTree` objects)

		:param regime: The category of the machine learning algorithm,
			e.g. supervised or RL
		:type regime: str

		:ivar gamma: The discount factor used to calculate returns.
			Only relevant for the RL regime. 
		:vartype gamma: float

		:ivar normalize_returns: Whether to normalize returns to be
			in the interval [0,1]. Only relevant for the RL regime
		:vartype normalize_returns: bool

		:ivar min_return: The minimum possible return. Used 
			if normalize returns==True. Only relevant for the RL regime
		:vartype min_return: float

		:ivar max_return: The maximum possible return. Used 
			if normalize returns==True. Only relevant for the RL regime
		:vartype max_return: float

		"""
		self.dataset = dataset
		self.model = model
		self.parse_trees = parse_trees
		self.regime = regime

		if self.regime == 'RL':
			self.gamma = kwargs['gamma']
			if kwargs['normalize_returns']==True:
				self.normalize_returns=True
				self.min_return = kwargs['min_return']
				self.max_return = kwargs['max_return']
			else:
				self.normalize_returns=False

	def run(self,candidate_solution,bound_method='ttest',**kwargs):
		""" Loop over parse trees, calculate the bounds on leaf nodes
		and propagate to the root node. The safety test passes if
		the upper bounds of all parse tree root nodes are less than or equal to 0. 

		:param candidate_solution: 
			The solution found by candidate selection
		:type candidate_solution: numpy ndarray

		:param bound_method: 
			The statistical method for calculating the confidence bounds
		:type bound_method: str, defaults to 'ttest'

		:return: passed, whether the candidate solution passed the safety test
		:rtype: bool
		
		"""
		passed = True
		for tree_i,pt in enumerate(self.parse_trees): 
			# before we propagate reset the tree
			pt.reset_base_node_dict()

			bounds_kwargs = dict(
				theta=candidate_solution,
				dataset=self.dataset,
				model=self.model,
				branch='safety_test',
				bound_method=bound_method,
				regime=self.regime
				)
			
			if self.regime == 'RL':
				bounds_kwargs['gamma'] = self.gamma
				bounds_kwargs['normalize_returns'] = self.normalize_returns
				if self.normalize_returns:
					bounds_kwargs['min_return'] = self.min_return
					bounds_kwargs['max_return'] = self.max_return

			pt.propagate_bounds(**bounds_kwargs)

			# Check if the i-th behavioral constraint is satisfied
			upperBound = pt.root.upper 
			if upperBound > 0.0: # If the current constraint was not satisfied, the safety test failed
				passed = False
			# reset bounds and data for each base node
			pt.reset_base_node_dict(reset_data=True)
		     
		return passed
