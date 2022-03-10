from operator import itemgetter
from functools import reduce

from .stats_utils import *
from src.constraints import *


class Node(object):
	""" 
	The base class for all parse tree nodes
	
	Attributes
	----------
	name : str
		the name of the node
	index : int
		the index of the node in the tree, root index is 0
	left : Node object or None
		left child node
	right : Node object or None
		right child node
	lower : float
		lower confidence bound
	upper : float
		upper confidence bound
	**kwargs : 
		optional additional arguments which get bundled into a dict

	Methods
	-------
	__repr__()
		String representation of the object 

	"""
	def __init__(self,name,lower,upper,**kwargs):
		self.name = name
		self.index = None 
		self.left  = None 
		self.right = None 
		self.lower = lower 
		self.upper = upper 

	def __repr__(self):
		lower_bracket = '['
		upper_bracket = ']'
		if np.isinf(self.lower):
			lower_bracket = '('
		if np.isinf(self.upper):
			upper_bracket = ')'

		bounds_str = f'{lower_bracket}{self.lower:g}, {self.upper:g}{upper_bracket}' \
			if (self.lower or self.upper) else '()'

		return '\n'.join(
			[
				'['+str(self.index)+']',
				str(self.name),
				u'\u03B5' + ' ' + bounds_str
			]
		) 
  

class BaseNode(Node):
	""" 
	Class for base variable leaf nodes
	in the parse tree.
	Inherits all attributes from Node class

	
	Attributes
	----------
	name : str
		The name of the node
	node_type : str
		equal to 'base_node'
	lower : float
		Lower confidence bound
	upper : float
		Upper confidence bound
	delta : float
		The share of the confidence put into this node
	compute_lower : bool
		Whether to compute the lower confidence interval
	compute_upper : bool
		Whether to compute the upper confidence interval
	conditional_columns: List(str)
		When calculating confidence bounds on a special 
		function, condition on these columns being == 1

	Methods
	-------
	calculate_bounds(bound_method)
		Calculate confidence bounds given a method, such as t-test
	
	compute_HC_lowerbound()
		--TODO--
		Calculate high confidence lower bound. 

	compute_HC_upperbound()
		--TODO--
		Calculate high confidence upper bound
	
	compute_HC_upper_and_lowerbound()
		--TODO--
		Calculate high confidence upper and lower bound

	"""
	def __init__(self,
		name,
		lower=float('-inf'),
		upper=float('inf'),
		conditional_columns=[],
		**kwargs):
		"""
		Parameters
		----------
		name : str
			The name of the node
		lower : float
			The lower bound, default -infinity
		upper : float
			The upper bound, default infinity
		"""

		super().__init__(name,lower,upper,**kwargs)
		self.node_type = 'base_node'
		self.delta = 0  
		self.measure_function_name = '' # non-empty if is_special True
		self.compute_lower = True 
		self.compute_upper = True 
		self.conditional_columns = conditional_columns

	def __repr__(self):
		""" 
		Overrides Node.__repr__()
		"""
		return super().__repr__() + ', ' + u'\u03B4' + f'={self.delta:g}'
	
	def mask_dataframe(self,dataset,conditional_columns):
		"""
		"""
		masks = reduce(np.logical_and,(dataset.df[col]==1 for col in conditional_columns))
		masked_df = dataset.df.loc[masks] 
		return masked_df

	def calculate_data_forbound(self,**kwargs):
		theta,dataset,model = itemgetter(
					'theta','dataset','model')(kwargs)

		if kwargs['branch'] == 'candidate_selection':
			# Then we're in candidate selection
			n_safety = kwargs['n_safety']

		# mask the data using the conditional columns, if present
		if self.conditional_columns:
			dataframe = self.mask_dataframe(
				dataset,self.conditional_columns)
		else:
			dataframe = dataset.df

		# If in candidate selection want to use safety data size
		# in bound calculation
		if kwargs['branch'] == 'candidate_selection':
			frac_masked = len(dataframe)/len(dataset.df)
			datasize = int(round(frac_masked*n_safety))
		else:
			datasize = len(dataframe)
		# Separate features from label
		label_column = dataset.label_column
		labels = dataframe[label_column]
		features = dataframe.loc[:, dataframe.columns != label_column]
		features.insert(0,'offset',1.0) # inserts a column of 1's

		# drop sensitive column names
		if dataset.sensitive_column_names:
			features = features.drop(columns=dataset.sensitive_column_names)
		data_dict = {'features':features,'labels':labels}  
		return data_dict,datasize

	def calculate_bounds(self,
		**kwargs):
		"""
		Parameters
		----------
		method : str
			The method for calculating the bounds, 
			default Student's t-test
		""" 
		if 'bound_method' in kwargs:
			bound_method = kwargs['bound_method']
			if bound_method == 'manual':
				# Bounds set by user
				return self.lower,self.upper

			elif bound_method == 'random':
				# Randomly assign lower and upper bounds
				lower, upper = (
					np.random.randint(0,2),
					np.random.randint(2,4)
					)
				return lower,upper
		
			else:
				# Real confidence bound 

				# --TODO-- abstract away to support things like 
				# getting confidence intervals from bootstrap
				# and RL cases
				branch = kwargs['branch']
				model = kwargs['model']
				theta = kwargs['theta']
				data_dict = kwargs['data_dict']

				# if user has custom ghat, this is where
				# we sample from it
				# if kwargs['']
				# take samples from the special function   
				estimator_samples = model.sample_from_statistic(
					self.measure_function_name,
					model,
					theta,
					data_dict)

				if self.compute_lower and self.compute_upper:
					if branch == 'candidate_selection':
						lower,upper = self.predict_HC_upper_and_lowerbound(
							data=estimator_samples,
							delta=self.delta,
							**kwargs)  
					elif branch == 'safety_test':
						lower,upper = self.compute_HC_upper_and_lowerbound(
							data=estimator_samples,
							delta=self.delta,
							**kwargs)  
					return lower,upper
				
				elif self.compute_lower:
					if branch == 'candidate_selection':
						lower = self.predict_HC_lowerbound(
							data=estimator_samples,
							delta=self.delta,
							**kwargs)  
					elif branch == 'safety_test':
						lower = self.compute_HC_lowerbound(
							data=estimator_samples,
							delta=self.delta,
							**kwargs)  
					return lower

				elif self.compute_upper:
					if branch == 'candidate_selection':
						upper = self.predict_HC_upperbound(
							data=estimator_samples,
							delta=self.delta,
							**kwargs)  
					elif branch == 'safety_test':
						upper = self.compute_HC_upperbound(
							data=estimator_samples,
							delta=self.delta,
							**kwargs)  
						return upper

				raise AssertionError("compute_lower and computer_upper cannot both be False")

		

	def compute_HC_lowerbound(self,
		data,
		datasize,
		delta,
		**kwargs):
		"""
		Parameters
		----------
		-- TODO -- 
		""" 
		if 'bound_method' in kwargs:
			bound_method = kwargs['bound_method']
			if bound_method == 'ttest':	
				lower = data.mean() - stddev(data) / np.sqrt(datasize) * tinv(1.0 - delta, datasize - 1)
			else:
				raise NotImplementedError(f"Bounding method {bound_method} is not supported yet")
			
		return lower

	def compute_HC_upperbound(self,
		data,
		datasize,
		delta,
		**kwargs):
		"""
		Parameters
		----------
		-- TODO -- 
		"""
		if 'bound_method' in kwargs:
			bound_method = kwargs['bound_method']
			if bound_method == 'ttest':
				upper = data.mean() + stddev(data) / np.sqrt(datasize) \
					* tinv(1.0 - delta, datasize - 1)
			else:
				raise NotImplementedError("Have not implemented" 
					f"confidence bounds with bound_method: {bound_method}")
			
		return upper
	
	def compute_HC_upper_and_lowerbound(self,
		data,
		datasize,
		delta,
		**kwargs):
		"""
		Parameters
		----------
		-- TODO -- 
		"""
		if 'bound_method' in kwargs:
			bound_method = kwargs['bound_method']
			if bound_method == 'ttest':
				lower = self.compute_HC_lowerbound(data=data,
					datasize=datasize,delta=delta/2,
					**kwargs)
				upper = self.compute_HC_upperbound(data=data,
					datasize=datasize,delta=delta/2,
					**kwargs)

			elif bound_method == 'manual':
				pass
			else:
				raise NotImplementedError("Have not implemented" 
					f"confidence bounds with bound_method: {bound_method}")
		else:
			raise NotImplementedError("Have not implemented" 
					"confidence bounds without the keyword bound_method")

		return lower,upper
  
	def predict_HC_lowerbound(self,
		data,
		datasize,
		delta,
		**kwargs):
		"""
		Parameters
		----------
		-- TODO -- 
		""" 
		if 'bound_method' in kwargs:
			bound_method = kwargs['bound_method']

			if bound_method == 'ttest':
				lower = data.mean() - 2*stddev(data) / np.sqrt(datasize) * tinv(1.0 - delta, datasize - 1)
			else:
				raise NotImplementedError(f"Bounding method {bound_method} is not supported yet")
			
		return lower

	def predict_HC_upperbound(self,
		data,
		datasize,
		delta,
		**kwargs):
		"""
		Parameters
		----------
		-- TODO -- 
		""" 
		if 'bound_method' in kwargs:
			bound_method = kwargs['bound_method']
			if bound_method == 'ttest':
				lower = data.mean() + 2*stddev(data) / np.sqrt(datasize) * tinv(1.0 - delta, datasize - 1)
			else:
				raise NotImplementedError(f"Bounding method {bound_method} is not supported yet")
			
		return lower

	def predict_HC_upper_and_lowerbound(self,
		data,
		datasize,
		delta,
		conditional_columns=[],
		**kwargs):
		"""
		Parameters
		----------
		-- TODO -- 
		"""
		if 'bound_method' in kwargs:
			bound_method = kwargs['bound_method']
			if bound_method == 'ttest':
				lower = self.predict_HC_lowerbound(data=data,
					datasize=datasize,delta=delta/2,
					**kwargs)
				upper = self.predict_HC_upperbound(data=data,
					datasize=datasize,delta=delta/2,
					**kwargs)

			elif bound_method == 'manual':
				pass
			else:
				raise NotImplementedError(f"Have not implemented" 
					"confidence bounds with bound_method: {bound_method}")
			
		return lower,upper


class CustomBaseNode(BaseNode):
	def __init__(self,
		name,
		lower=float('-inf'),
		upper=float('inf'),
		**kwargs):
		"""
		Parameters
		----------
		name : str
			The name of the node
		lower : float
			The lower bound, default -infinity
		upper : float
			The upper bound, default infinity
		"""
		super().__init__(name,lower,upper,**kwargs)
		self.node_type = 'custom_base_node'
		self.delta = 0  

		if 'epsilon' in kwargs:
			self.epsilon = kwargs['epsilon']
		else:
			self.epsilon = None

		if 'compute_lower' in kwargs:
			self.compute_lower = kwargs['compute_lower'] 
		else:
			self.compute_lower = True

		if 'compute_upper' in kwargs:
			self.compute_upper = kwargs['compute_upper'] 
		else:
			self.compute_upper = True

	def calculate_data_forbound(self,**kwargs):
		""" Overrides parent method """
		dataset = kwargs['dataset']
		dataframe = dataset.df

		# Use custom ghat dict to find the ghat function
		custom_dict_entry = custom_ghat_dict[self.name]
		ghat_class = custom_dict_entry['class']
		method_str = custom_dict_entry['method']
		# Make class instance
		ghat_instance = ghat_class(
			str_expression=self.name,
			epsilon=self.epsilon)
		
		# call the method to precalculate data
		label_column = dataset.label_column
		labels = dataframe[label_column]
		features = dataframe.loc[:, dataframe.columns != label_column]
		features.insert(0,'offset',1.0) # inserts a column of 1's
		
		# Do not drop the sensitive columns yet. 
		# They might be needed in precalculate_data()

		data_dict,datasize = ghat_instance.precalculate_data(
			features,labels,**kwargs)

		if kwargs['branch'] == 'candidate_selection':
			# print("figuring out fraction of safety data to take")
			n_safety = kwargs['n_safety']
			# frac_masked = len(dataframe)/len(dataset.df)
			frac_masked = datasize/len(dataframe)
			datasize = int(round(frac_masked*n_safety))

		return data_dict,datasize

	def calculate_bounds(self,
		**kwargs):
		"""
		Overrides parent method
		""" 
		# print("in calculate_bounds()")
		if 'bound_method' in kwargs:
			bound_method = kwargs['bound_method']
			if bound_method == 'manual':
				# Bounds set by user
				return self.lower,self.upper

			elif bound_method == 'random':
				# Randomly assign lower and upper bounds
				lower, upper = (
					np.random.randint(0,2),
					np.random.randint(2,4)
					)
				return lower,upper
		
			else:
				# Real confidence bound 
				branch = kwargs['branch']
				model = kwargs['model']
				theta = kwargs['theta']
				data_dict = kwargs['data_dict']
				# Use custom ghat dict to find the ghat function
				custom_dict_entry = custom_ghat_dict[self.name]
				ghat_class = custom_dict_entry['class']
				method_str = custom_dict_entry['method']
				# Make class instance
				ghat_instance = ghat_class(
					str_expression=self.name,
					epsilon=self.epsilon)
				# call the method to generate the samples of ghat
				ghat_method = getattr(ghat_instance,method_str)
				estimator_samples = ghat_method(model,theta,data_dict)
				# print("estimator_samples: ",estimator_samples)
				# print(estimator_samples.mean())
				if self.compute_lower and self.compute_upper:
					if branch == 'candidate_selection':
						lower,upper = self.predict_HC_upper_and_lowerbound(
							data=estimator_samples,
							delta=self.delta,
							**kwargs)  
					elif branch == 'safety_test':
						lower,upper = self.compute_HC_upper_and_lowerbound(
							data=estimator_samples,
							delta=self.delta,
							**kwargs)  
					return lower,upper
				
				elif self.compute_lower:
					if branch == 'candidate_selection':
						lower = self.predict_HC_lowerbound(
							data=estimator_samples,
							delta=self.delta,
							**kwargs)  
					elif branch == 'safety_test':
						lower = self.compute_HC_lowerbound(
							data=estimator_samples,
							delta=self.delta,
							**kwargs)  
					return lower,lower

				elif self.compute_upper:
					if branch == 'candidate_selection':
						upper = self.predict_HC_upperbound(
							data=estimator_samples,
							delta=self.delta,
							**kwargs)  
					elif branch == 'safety_test':
						upper = self.compute_HC_upperbound(
							data=estimator_samples,
							delta=self.delta,
							**kwargs)  
					return upper,upper

				raise AssertionError("compute_lower and compute_upper cannot both be False")


class ConstantNode(Node):
	""" 
	Class for constant leaf nodes 
	in the parse tree. 
	Inherits all attributes from Node class

	Attributes
	----------
	name : str
		The name of the node
	value: float
		The value of the constant the node represents
	node_type : str
		'constant_node'

	"""
	def __init__(self,name,value,**kwargs):
		"""
		Sets lower and upper bound as the value of 
		the constant

		Parameters
		----------
		name : str
			The name of the node
		value: float
			The value of the constant 
		"""
		super().__init__(name=name,
			lower=value,upper=value,**kwargs)
		self.value = value
		self.node_type = 'constant_node'
  

class InternalNode(Node):
	""" 
	Class for internal (non-leaf) nodes 
	in the parse tree.
	These represent operators, such as +,-,*,/ etc.
	Inherits all attributes from Node class

	Attributes
	----------
	name : str
		The name of the node
	"""
	def __init__(self,name,
		lower=float('-inf'),upper=float('inf'),**kwargs):
		super().__init__(name,lower,upper,**kwargs)
		self.node_type = 'internal_node'
