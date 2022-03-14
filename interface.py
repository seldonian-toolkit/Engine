import sys
from src.parse_tree import ParseTree


if __name__ == '__main__':
	# constraint_str = input("Enter your constraint: ")
	# constraint_str = 'abs((Mean_Error | [M]) - (Mean_Error | [F])) - 0.1'
	# constraint_str = 'X**Y - 2.0'
	# constraint_str = '(Mean_Error | [M]) - 0.1'
	# constraint_str = 'FPR | [M] - FNR | [F] - 0.1'
	# constraint_str = 'F1**3 + FPR**3'
	constraint_str = '(FPR + FNR) - 0.25'
	delta = 0.05

	# Create parse tree object
	parse_tree = ParseTree(delta=delta)

	# Fill out tree
	parse_tree.create_from_ast(constraint_str)
	
	# assign deltas for each base node
	# use equal weighting for each base node
	parse_tree.assign_deltas(weight_method='equal')

	# Assign bounds needed on the base nodes
	# parse_tree.assign_bounds_needed()
	# print(parse_tree.root.left.compute_lower)
	# print(parse_tree.root.left.compute_upper)
	# # # # Propagate bounds using random interval assignment to base variables
	# parse_tree.propagate_bounds(bound_method='random')

	# # # Create the graphviz visualization and render it to a PDF
	title = f'Parse tree for expression:\n{constraint_str}\ndelta={delta}'
	graph = parse_tree.make_viz(title)
	graph.attr(fontsize='12')
	graph.view() # to open it as a pdf
