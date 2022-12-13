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
	all_col_names = metadata_dict['all_col_names']
				
	loader_classification = DataSetLoader(
		regime=regime)

	# Load dataset from file
		
	# First, from csv
	data_pth_csv = 'static/datasets/supervised/GPA/gpa_classification_dataset.csv'

	dataset_fromcsv = loader_classification.load_supervised_dataset(
		filename=data_pth_csv,
		metadata_filename=metadata_pth,
		file_type='csv')
	assert dataset_fromcsv.meta_information == {
		'feature_col_names': ['Physics', 'Biology', 'History', 'Second_Language', 'Geography', 'Literature', 'Portuguese_and_Essay', 'Math', 'Chemistry'],
		'label_col_names': ['GPA_class'],
		'sensitive_col_names': ['M', 'F'], 'sub_regime': 'classification'}
	assert dataset_fromcsv.features.shape == (43303,9)
	assert dataset_fromcsv.sensitive_col_names == ["M","F"]

	# Try loading a bogus file type
	data_pth_txt = 'static/datasets/supervised/GPA/gpa_classification_dataset.txt'

	with pytest.raises(NotImplementedError) as excinfo:
		dataset_fromtxt = loader_classification.load_supervised_dataset(
			filename=data_pth_txt,
			metadata_filename=metadata_pth,
			file_type='txt')
	error_str = f"File type: txt not supported"
	assert error_str in str(excinfo.value)


	# Regression 
	metadata_pth = 'static/datasets/supervised/GPA/metadata_regression.json'
	metadata_dict = load_json(metadata_pth)
	regime = metadata_dict['regime']
	sub_regime = metadata_dict['sub_regime']
	all_col_names = metadata_dict['all_col_names']
			

	loader_regression = DataSetLoader(
		regime=regime)

	# Load dataset from file
		
	# First, from csv
	data_pth_csv = 'static/datasets/supervised/GPA/gpa_regression_dataset.csv'

	dataset_fromcsv = loader_regression.load_supervised_dataset(
		filename=data_pth_csv,
		metadata_filename=metadata_pth,
		file_type='csv')

	assert dataset_fromcsv.meta_information == {
		'feature_col_names': ['Physics', 'Biology', 'History', 'Second_Language', 'Geography', 'Literature', 'Portuguese_and_Essay', 'Math', 'Chemistry'],
		'label_col_names': ['GPA'],
		'sensitive_col_names': ['M', 'F'], 'sub_regime': 'regression'}
	assert dataset_fromcsv.features.shape == (43303,9)
	assert dataset_fromcsv.sensitive_col_names == ["M","F"]

	# Make sure error is raised if labels or sensitive attributes are passed as lists
	
	features = dataset_fromcsv.features
	listlabels = list(dataset_fromcsv.labels)
	sensitive_attrs = dataset_fromcsv.sensitive_attrs
	meta_information = dataset_fromcsv.meta_information
	with pytest.raises(AssertionError) as excinfo:
		ds = SupervisedDataSet(
			features=features,
			labels=listlabels,
			sensitive_attrs=sensitive_attrs,
			num_datapoints=len(features),
			meta_information=meta_information)
	error_str = "labels must be a numpy array"
	assert str(excinfo.value) == error_str

	features = dataset_fromcsv.features
	labels = dataset_fromcsv.labels
	listsensitive_attrs = [dataset_fromcsv.sensitive_attrs]
	meta_information = dataset_fromcsv.meta_information
	with pytest.raises(AssertionError) as excinfo:
		ds = SupervisedDataSet(
			features=features,
			labels=labels,
			sensitive_attrs=listsensitive_attrs,
			num_datapoints=len(features),
			meta_information=meta_information)
	error_str = "sensitive_attrs must be a numpy array or []"
	assert str(excinfo.value) == error_str

	# But allow sensitive attributes to be an empty list

	ds = SupervisedDataSet(
		features=features,
		labels=labels,
		sensitive_attrs=[],
		num_datapoints=len(features),
		meta_information=meta_information)

	assert ds.sensitive_attrs == []

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

	assert dataset_fromcsv.meta_information['episode_col_names'] == columns
	assert dataset_fromcsv.meta_information['sensitive_col_names'] == []
	episodes = dataset_fromcsv.episodes
	assert len(episodes) == 100
	assert np.allclose(episodes[0].observations[0:5],np.array([0,0,1,4,5]))
	assert np.allclose(episodes[0].actions[0:5],np.array([0,1,2,1,0]))
	assert np.allclose(episodes[0].rewards[0:5],np.array([0,0,0,0,0]))
	assert np.allclose(episodes[0].action_probs[0:5],np.array([0.25,0.25,0.25,0.25,0.25]))

	# Now from pickled episode list
	data_pth_pkl = 'static/datasets/RL/gridworld/gridworld_100episodes.pkl'
	
	dataset_frompkl = loader.load_RL_dataset_from_episode_file(
		filename=data_pth_pkl)
	episodes = dataset_frompkl.episodes
	assert len(episodes) == 100
	assert np.allclose(episodes[0].observations[0:5],np.array([0,0,1,4,5]))
	assert np.allclose(episodes[0].actions[0:5],np.array([0,1,2,1,3]))
	assert np.allclose(episodes[0].rewards[0:5],np.array([0,0,0,0,0]))
	assert np.allclose(episodes[0].action_probs[0:5],np.array([0.25,0.25,0.25,0.25,0.25]))
