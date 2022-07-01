import pytest
import importlib
import autograd.numpy as np

from seldonian.utils.io_utils import load_json, load_pickle
from seldonian.parse_tree.parse_tree import ParseTree
from seldonian.dataset import (DataSetLoader,
SupervisedDataSet,RLDataSet)
from seldonian.models.model import (LinearRegressionModel,
TabularSoftmaxModel)
from seldonian.spec import RLSpec, SupervisedSpec
from seldonian.seldonian_algorithm import seldonian_algorithm



### Begin tests

def test_load_supervised_dataset():
	""" Test that supervised learning datasets can be loaded
	from various formats """

	# Classification 
	metadata_pth = 'static/datasets/supervised/GPA/metadata_classification.json'
	metadata_dict = load_json(metadata_pth)
	regime = metadata_dict['regime']
	sub_regime = metadata_dict['sub_regime']
	columns = metadata_dict['columns']
				
	include_sensitive_columns = False
	include_intercept_term = True

	loader_classification = DataSetLoader(
		regime=regime)

	# Load dataset from file
		
	# First, from csv
	data_pth_csv = 'static/datasets/supervised/GPA/gpa_classification_dataset.csv'

	dataset_fromcsv = loader_classification.load_supervised_dataset(
		filename=data_pth_csv,
		metadata_filename=metadata_pth,
		include_sensitive_columns=include_sensitive_columns,
		include_intercept_term=include_intercept_term,
		file_type='csv')

	assert dataset_fromcsv.meta_information == columns
	assert dataset_fromcsv.label_column == "GPA_class"
	assert len(dataset_fromcsv.df) == 43303
	assert dataset_fromcsv.sensitive_column_names == ["M","F"]
	assert dataset_fromcsv.include_sensitive_columns == False
	assert dataset_fromcsv.include_intercept_term == True

	# Now from pickle
	data_pth_pkl = 'static/datasets/supervised/GPA/gpa_classification_dataset.pkl'

	dataset_frompkl = loader_classification.load_supervised_dataset(
		filename=data_pth_pkl,
		metadata_filename=metadata_pth,
		include_sensitive_columns=include_sensitive_columns,
		include_intercept_term=include_intercept_term,
		file_type='pkl')

	assert dataset_frompkl.meta_information == columns
	assert dataset_frompkl.label_column == "GPA_class"
	assert len(dataset_frompkl.df) == 43303
	assert dataset_frompkl.sensitive_column_names == ["M","F"]
	assert dataset_frompkl.include_sensitive_columns == False
	assert dataset_frompkl.include_intercept_term == True

	# Try loading a bogus file type

	with pytest.raises(NotImplementedError) as excinfo:
		dataset_frompkl = loader_classification.load_supervised_dataset(
			filename=data_pth_pkl,
			metadata_filename=metadata_pth,
			include_sensitive_columns=include_sensitive_columns,
			include_intercept_term=include_intercept_term,
			file_type='txt')
	error_str = f"File type: txt not supported"
	assert error_str in str(excinfo.value)


	# Regression 
	metadata_pth = 'static/datasets/supervised/GPA/metadata_regression.json'
	metadata_dict = load_json(metadata_pth)
	regime = metadata_dict['regime']
	sub_regime = metadata_dict['sub_regime']
	columns = metadata_dict['columns']
				
	include_sensitive_columns = False
	include_intercept_term = True

	loader_regression = DataSetLoader(
		regime=regime)

	# Load dataset from file
		
	# First, from csv
	data_pth_csv = 'static/datasets/supervised/GPA/gpa_regression_dataset.csv'

	dataset_fromcsv = loader_regression.load_supervised_dataset(
		filename=data_pth_csv,
		metadata_filename=metadata_pth,
		include_sensitive_columns=include_sensitive_columns,
		include_intercept_term=include_intercept_term,
		file_type='csv')

	assert dataset_fromcsv.meta_information == columns
	assert dataset_fromcsv.label_column == "GPA"
	assert len(dataset_fromcsv.df) == 43303
	assert dataset_fromcsv.sensitive_column_names == ["M","F"]
	assert dataset_fromcsv.include_sensitive_columns == False
	assert dataset_fromcsv.include_intercept_term == True

	# Now from pickle
	data_pth_pkl = 'static/datasets/supervised/GPA/gpa_regression_dataset.pkl'

	dataset_frompkl = loader_regression.load_supervised_dataset(
		filename=data_pth_pkl,
		metadata_filename=metadata_pth,
		include_sensitive_columns=include_sensitive_columns,
		include_intercept_term=include_intercept_term,
		file_type='pkl')

	assert dataset_frompkl.meta_information == columns
	assert dataset_frompkl.label_column == "GPA"
	assert len(dataset_frompkl.df) == 43303
	assert dataset_frompkl.sensitive_column_names == ["M","F"]
	assert dataset_frompkl.include_sensitive_columns == False
	assert dataset_frompkl.include_intercept_term == True

def test_load_RL_dataset():
	""" Test that reinforcement learning datasets can be loaded
	from various formats """

	# Classification 
	metadata_pth = 'static/datasets/RL/gridworld/gridworld3x3_metadata.json'
	metadata_dict = load_json(metadata_pth)
	regime = metadata_dict['regime']
	columns = metadata_dict['columns']
	RL_environment_name = metadata_dict['RL_environment_name']

	loader = DataSetLoader(
		regime=regime)

	# Load dataset from file
		
	# First, from csv
	data_pth_csv = 'static/datasets/RL/gridworld/gridworld3x3_1000episodes.csv'

	dataset_fromcsv = loader.load_RL_dataset_from_csv(
		filename=data_pth_csv,
		metadata_filename=metadata_pth)

	assert dataset_fromcsv.meta_information == columns
	episodes = dataset_fromcsv.episodes
	assert len(episodes) == 1000
	assert np.allclose(episodes[0].states,np.array([0,0,3,4,7]))
	assert np.allclose(episodes[0].actions,np.array([2,1,3,1,3]))
	assert np.allclose(episodes[0].rewards,np.array([0,0,0,-1,1]))
	assert np.allclose(episodes[0].pis,np.array([0.25,0.25,0.25,0.25,0.25]))

	# Now from pickled dataframe
	# First, from csv
	data_pth_dataframe = 'static/datasets/RL/gridworld/gridworld3x3_1000episodes_dataframe.pkl'

	dataset_fromdf = loader.load_RL_dataset_from_dataframe(
		filename=data_pth_dataframe,
		metadata_filename=metadata_pth)


	assert dataset_fromdf.meta_information == columns
	episodes = dataset_fromdf.episodes
	assert len(episodes) == 1000
	assert np.allclose(episodes[0].states,np.array([0,0,3,4,7]))
	assert np.allclose(episodes[0].actions,np.array([2,1,3,1,3]))
	assert np.allclose(episodes[0].rewards,np.array([0,0,0,-1,1]))
	assert np.allclose(episodes[0].pis,np.array([0.25,0.25,0.25,0.25,0.25]))

	# Now from pickled episode list
	data_pth_episode_list = 'static/datasets/RL/gridworld/gridworld3x3_250episodes_list.pkl'

	dataset_from_episode_list = loader.load_RL_dataset_from_episode_list(
		filename=data_pth_episode_list,
		metadata_filename=metadata_pth)

	assert dataset_from_episode_list.meta_information == columns
	episodes = dataset_from_episode_list.episodes
	assert len(episodes) == 250

	assert np.allclose(episodes[0].states,
		np.array([[0,0,0,0,1,2,2,2,2,5,5]]))

	assert np.allclose(episodes[0].actions,
		np.array([[0,2,2,3,3,3,3,0,1,3,1]]))

	assert np.allclose(episodes[0].rewards,
		np.array([0,0,0,0,0,0,0,0,0,0,1]))
	assert np.allclose(episodes[0].pis,
		np.array([0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25]))	
	
