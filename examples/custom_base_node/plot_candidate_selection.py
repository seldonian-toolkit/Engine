from seldonian.utils.plot_utils import plot_gradient_descent
from seldonian.utils.io_utils import load_pickle

def main():
	f = "/Users/ahoag/beri/code/engine-repo/examples/custom_base_node/logs/candidate_selection_log20.p"
	solution_dict = load_pickle(f)
	fig = plot_gradient_descent(
		solution_dict,
		primary_objective_name='Mean Squared Error',
		save=True,
		savename='custom_base_node_candidate_selection.png')

if __name__ == "__main__":
	main()