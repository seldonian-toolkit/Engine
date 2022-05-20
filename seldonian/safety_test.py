import autograd.numpy as np   # Thinly-wrapped version of Numpy

class SafetyTest(object):
	""" 
	Class for the safety test
	
	Attributes
	----------
	dataset : src.dataset.DataSet instance
		Object representing the safety dataset
	model   : src.model.SeldonianModel (or child) instance
		Object representing the model
	parse_trees : List(src.parse_tree.Parse_Tree)
		List of parse trees containing the behavioral consraints

	Methods
	-------
	run(candidate_solution,bound_method)
		Run the safety test given a candidate solution
		and a method for computing the confidence bounds

	"""
	def __init__(self,dataset,model,parse_trees,regime='supervised',**kwargs):
		self.dataset = dataset
		self.model = model
		self.parse_trees = parse_trees
		self.regime = regime

		if self.regime == 'RL':
			self.gamma = kwargs['gamma']
			self.min_return = kwargs['min_return']
			self.max_return = kwargs['max_return']

	def run(self,candidate_solution,bound_method='ttest',**kwargs):
		# Loop over parse trees and propagate
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
				bounds_kwargs['min_return'] = self.min_return
				bounds_kwargs['max_return'] = self.max_return

			pt.propagate_bounds(**bounds_kwargs)

			# graph = pt.make_viz(pt.constraint_str)
			# graph.view()
			# input("next")
			# Check if the i-th behavioral constraint is satisfied
			upperBound = pt.root.upper 
			print(f"Upper bound from running safety test: {upperBound}") 
			if upperBound > 0.0: # If the current constraint was not satisfied, the safety test failed
				passed = False
			# reset bounds and data for this node
			pt.reset_base_node_dict(reset_data=True)
		
		# If we get here, all of the behavioral constraints were satisfied      
		return passed
