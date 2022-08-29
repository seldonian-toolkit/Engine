import pytest
import importlib
import autograd.numpy as np

from seldonian.utils.io_utils import load_json
from seldonian.dataset import (DataSetLoader,
SupervisedDataSet,RLDataSet)

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
	metadata_pth = 'static/datasets/RL/gridworld/gridworld_metadata.json'
	metadata_dict = load_json(metadata_pth)
	regime = metadata_dict['regime']
	columns = metadata_dict['columns']

	loader = DataSetLoader(
		regime=regime)

	# Load dataset from file
		
	# First, from csv
	data_pth_csv = 'static/datasets/RL/gridworld/gridworld_100episodes.csv'

	dataset_fromcsv = loader.load_RL_dataset_from_csv(
		filename=data_pth_csv,
		metadata_filename=metadata_pth)

	assert dataset_fromcsv.meta_information == columns
	episodes = dataset_fromcsv.episodes
	assert len(episodes) == 100
	assert np.allclose(episodes[0].observations[0:5],np.array([0,0,1,4,5]))
	assert np.allclose(episodes[0].actions[0:5],np.array([0,1,2,1,0]))
	assert np.allclose(episodes[0].rewards[0:5],np.array([0,0,0,0,0]))
	assert np.allclose(episodes[0].pis[0:5],np.array([0.25,0.25,0.25,0.25,0.25]))

	# Now from pickled episode list
	data_pth_pkl = 'static/datasets/RL/gridworld/gridworld_100episodes.pkl'
	
	dataset_frompkl = loader.load_RL_dataset_from_episode_file(
		filename=data_pth_pkl)
	episodes = dataset_frompkl.episodes
	assert len(episodes) == 100
	assert np.allclose(episodes[0].observations[0:5],np.array([0,0,1,4,5]))
	assert np.allclose(episodes[0].actions[0:5],np.array([0,1,2,1,0]))
	assert np.allclose(episodes[0].rewards[0:5],np.array([0,0,0,0,0]))
	assert np.allclose(episodes[0].pis[0:5],np.array([0.25,0.25,0.25,0.25,0.25]))
