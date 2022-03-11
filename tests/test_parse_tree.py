from src.parse_tree import *
from src.dataset import *
from src.safety_test import SafetyTest
import pytest
import time

two_interval_options = [
    [[2.0,3.0],[4.0,5.0]],
    [[-2.0,1.0],[2.0,3.0]],
    [[0.5,0.75],[2,4]],
    [[0.5,1.1],[0.0,0.4]],
    [[-3.2,-2.0],[-4.5,-4.0]],
    [[-3.2,-2.0],[-4.5,5.0]],
    [[-6.1,-6.0],[0.0,0.5]],
    [[0.0,0.5],[1.0,9.0]],
    [[0.0,0.5],[-6.8,-5.0]],
    [[float('-inf'),5.0],[-6.8,-5.0]],
    [[0.0,float('inf')],[5.0,10.0]],
    [[float('-inf'),float('inf')],[5.0,10.0]],
    [[float('-inf'),float('inf')],[float('-inf'),float('inf')]],
    [[float('inf'),float('inf')],[float('inf'),float('inf')]],
]

single_interval_options = [
    [-3.2,-2.0],
    [-3.2,2.0],
    [-5.1,0],
    [0.0,0.5],
    [float('-inf'),0.0],
    [float('-inf'),15342],
    [0.0,float('inf')],
    [float('-inf'),float('inf')],
    [float('inf'),float('inf')]
]

answer_dict = {
    'add': [
        [6.0,8.0],
        [0.0,4.0],
        [2.5,4.75],
        [0.5,1.5],
        [-7.7, -6.0],
        [-7.7,3.0],
        [-6.1, -5.5],
        [1.0, 9.5],
        [-6.8, -4.5],
        [float('-inf'), 0.0],
        [5.0, float('inf')],
        [float('-inf'), float('inf')],
        [float('-inf'), float('inf')],
        [float('inf'), float('inf')],
    ],
    'sub': [
        [-3.0,-1.0],
        [-5.0,-1.0],
        [-3.5,-1.25],
        [0.1,1.1],
        [0.8, 2.5],
        [-8.2, 2.5],
        [-6.6, -6.0],
        [-9.0, -0.5],
        [5.0, 7.3],
        [float('-inf'), 11.8],
        [-10.0, float('inf')],
        [float('-inf'), float('inf')],
        [float('-inf'), float('inf')],
        [float('-inf'), float('inf')],
    ],
    'mult': [
        [8.0,15.0],
        [-6.0,3.0],
        [1.0,3.0],
        [0.0,1.1*0.4],
        [8.0, 14.4],
        [-16.0, 14.4],
        [-3.05, 0.0],
        [0.0, 4.5],
        [-3.4, 0.0],
        [-34.0, float('inf')],
        [0.0, float('inf')],
        [float('-inf'), float('inf')],
        [float('-inf'), float('inf')],
        [float('inf'), float('inf')],
    ],

    'div': [
        [2/5.0,3/4.0],
        [-1.0,0.5],
        [1/8,3/8],
        [0.5/0.4,float('inf')],
        [2/4.5, 3.2/4],
        [float('-inf'),float('inf')],
        [float('-inf'), -12.0],
        [0.0, 0.5],
        [-0.1, 0.0],
        [-1.0, float('inf')],
        [0.0, float('inf')],
        [float('-inf'), float('inf')],
        [float('-inf'), float('inf')],
        [float('-inf'), float('inf')],
    ],
    'pow':[
        [16,243],
        [None,None],
        [pow(0.5,4),pow(0.75,2)],
        [pow(0.5,0.4),pow(1.1,0.4)],
        [None,None], # input raises exception 
        [None,None], # input raises exception 
        [None,None], # input raises exception 
        [0.0, 0.5],
        [0.0, 0.5],
        [None,None], # input raises exception 
        [0.0,float('inf')],
        [None,None], # input raises exception 
        [None,None], # input raises exception 
        [float('inf'),float('inf')]
    ],
    'min': [
        [2.0,3.0],
        [-2.0,1.0],
        [0.5,0.75],
        [0.0,0.4],
        [-4.5,-4.0],
        [-4.5,-2.0],
        [-6.1,-6.0],
        [0.0,0.5],
        [-6.8,-5.0],
        [float('-inf'),-5.0],
        [0.0,10.0],
        [float('-inf'),10.0],
        [float('-inf'),float('inf')],
        [float('inf'),float('inf')]
    ],
    'max': [
        [4.0,5.0],
        [2.0,3.0],
        [2.0,4.0],
        [0.5,1.1],
        [-3.2,-2.0],
        [-3.2,5.0],
        [0.0,0.5],
        [1.0,9.0],
        [0.0,0.5],
        [-6.8,5.0],
        [5.0,float('inf')],
        [5.0,float('inf')],
        [float('-inf'),float('inf')],
        [float('inf'),float('inf')]
    ],
    'abs': [
        [2.0,3.2],
        [0,3.2],
        [0,5.1],
        [0,0.5],
        [0,float('inf')],
        [0,float('inf')],
        [0,float('inf')],
        [0,float('inf')],
        [float('inf'),float('inf')]
    ]

}

### Begin tests

########################
### Propagator tests ###
########################


@pytest.mark.parametrize('interval_index',range(len(two_interval_options)))
def test_add_bounds(interval_index,stump):
    ### Addition ###

    a,b=two_interval_options[interval_index]
    answer = answer_dict['add'][interval_index]
    pt = stump('add',a,b)
    pt.propagate_bounds(bound_method='manual')
    assert pt.root.lower == answer[0]
    assert pt.root.upper == answer[1]
    assert pt.base_node_dict['a']['computed'] == True
    assert pt.base_node_dict['b']['computed'] == True

@pytest.mark.parametrize('interval_index',range(len(two_interval_options)))
def test_subtract_bounds(interval_index,stump):
    ### Subtraction ###

    a,b=two_interval_options[interval_index]
    answer = answer_dict['sub'][interval_index]
    pt = stump('sub',a,b)
    pt.propagate_bounds(bound_method='manual')
    # Use approx due to floating point imprecision
    assert pt.root.lower == pytest.approx(answer[0])
    assert pt.root.upper == pytest.approx(answer[1])
    assert pt.base_node_dict['a']['computed'] == True
    assert pt.base_node_dict['b']['computed'] == True

@pytest.mark.parametrize('interval_index',range(len(two_interval_options)))
def test_multiply_bounds(interval_index,stump):
    ### Multiplication ###

    a,b=two_interval_options[interval_index]
    answer = answer_dict['mult'][interval_index]
    pt = stump('mult',a,b)
    pt.propagate_bounds(bound_method='manual')
    # Use approx due to floating point imprecision
    assert pt.root.lower == pytest.approx(answer[0])
    assert pt.root.upper == pytest.approx(answer[1])
    assert pt.base_node_dict['a']['computed'] == True
    assert pt.base_node_dict['b']['computed'] == True

@pytest.mark.parametrize('interval_index',range(len(two_interval_options)))
def test_divide_bounds(interval_index,stump):
    ### Division ###

    a,b=two_interval_options[interval_index]
    answer = answer_dict['div'][interval_index]
    pt = stump('div',a,b)
    pt.propagate_bounds(bound_method='manual')
    # Use approx due to floating point imprecision
    assert pt.root.lower == pytest.approx(answer[0])
    assert pt.root.upper == pytest.approx(answer[1])
    assert pt.base_node_dict['a']['computed'] == True
    assert pt.base_node_dict['b']['computed'] == True

@pytest.mark.parametrize('interval_index',range(len(two_interval_options)))
def test_power_bounds(interval_index,stump):
    ### power ###

    a,b=two_interval_options[interval_index]
    print(a,b)
    pt = stump('pow',a,b)
    if a[0] < 0:
        with pytest.raises(ArithmeticError) as excinfo:
            pt.propagate_bounds(bound_method='manual')
        
        assert "Cannot compute interval" in str(excinfo.value)
        assert "because first argument contains negatives" in str(excinfo.value)

    elif 0 in a and (b[0]<0 or b[1]<1):
        with pytest.raises(ZeroDivisionError) as excinfo:
            pt.propagate_bounds(bound_method='manual')
        
        assert "0.0 cannot be raised to a negative power" in str(excinfo.value)
    else:
        answer = answer_dict['pow'][interval_index]
        print(answer)
        pt.propagate_bounds(bound_method='manual')
        # Use approx due to floating point imprecision
        assert pt.root.lower == pytest.approx(answer[0])
        assert pt.root.upper == pytest.approx(answer[1])
        assert pt.base_node_dict['a']['computed'] == True
        assert pt.base_node_dict['b']['computed'] == True

@pytest.mark.parametrize('interval_index',range(len(two_interval_options)))
def test_min_bounds(interval_index,stump):
    ### min ###

    a,b=two_interval_options[interval_index]
    print(a,b)
    pt = stump('min',a,b)
    
    answer = answer_dict['min'][interval_index]
    print(answer)
    pt.propagate_bounds(bound_method='manual')
    # Use approx due to floating point imprecision
    assert pt.root.lower == pytest.approx(answer[0])
    assert pt.root.upper == pytest.approx(answer[1])
    assert pt.base_node_dict['a']['computed'] == True
    assert pt.base_node_dict['b']['computed'] == True

@pytest.mark.parametrize('interval_index',range(len(two_interval_options)))
def test_max_bounds(interval_index,stump):
    ### min ###

    a,b=two_interval_options[interval_index]
    print(a,b)
    pt = stump('max',a,b)
    
    answer = answer_dict['max'][interval_index]
    print(answer)
    pt.propagate_bounds(bound_method='manual')
    # Use approx due to floating point imprecision
    assert pt.root.lower == pytest.approx(answer[0])
    assert pt.root.upper == pytest.approx(answer[1])
    assert pt.base_node_dict['a']['computed'] == True
    assert pt.base_node_dict['b']['computed'] == True


@pytest.mark.parametrize('interval_index',range(len(single_interval_options)))
def test_abs_bounds(interval_index,edge):
    ### Absolute value ###

    a=single_interval_options[interval_index]
    answer = answer_dict['abs'][interval_index]
    pt = edge('abs',a)
    pt.propagate_bounds(bound_method='manual')
    # Use approx due to floating point imprecision
    assert pt.root.lower == pytest.approx(answer[0])
    assert pt.root.upper == pytest.approx(answer[1])
    assert pt.base_node_dict['a']['computed'] == True

########################
### Parse tree tests ###
########################

def test_parse_tree_from_simple_string():
    constraint_str = 'FPR - (FNR + PR)*4'
    delta = 0.05
    pt = ParseTree(delta)
    pt.create_from_ast(constraint_str)
    assert pt.n_nodes == 7
    assert pt.n_base_nodes == 3
    assert len(pt.base_node_dict) == 3
    assert isinstance(pt.root,InternalNode)
    assert pt.root.name == 'sub'
    assert pt.root.left.name == 'FPR'
    assert pt.root.right.name == 'mult'
    assert pt.root.right.left.name == 'add'
    assert pt.root.right.left.left.name == 'FNR'
    assert pt.root.right.left.right.name == 'PR'
    assert pt.root.right.right.name == '4'
    assert pt.root.right.right.value == 4

def test_measure_functions_recognized():
    constraint_str = 'Mean_Squared_Error - 2.0'
    delta = 0.05 

    pt = ParseTree(delta)
    pt.create_from_ast(constraint_str)
    assert pt.root.left.measure_function_name == 'Mean_Squared_Error'
    
    constraint_str = '(Mean_Error|[M]) - 2.0'
    delta = 0.05 

    pt = ParseTree(delta)
    pt.create_from_ast(constraint_str)
    assert pt.root.left.measure_function_name == 'Mean_Error'

    # Test that a non-measure base node 
    # is not recognized as measure
    constraint_str = 'X - 2.0'
    delta = 0.05 

    pt = ParseTree(delta)
    with pytest.raises(NotImplementedError) as excinfo:
        pt.create_from_ast(constraint_str)
    
    error_str = ("Error parsing your expression."
             " A variable name was used which we do not recognize: X")
    assert str(excinfo.value) == error_str
    
def test_raise_error_on_excluded_operators():

    constraint_str = 'FPR^4'
    delta = 0.05
    pt = ParseTree(delta)
    with pytest.raises(NotImplementedError) as excinfo:
        pt.create_from_ast(constraint_str)
    error_str = ("Error parsing your expression."
         " An operator was used which we do not support: ^")
    assert str(excinfo.value) == error_str

    constraint_str = 'FPR<<4'
    delta = 0.05
    pt = ParseTree(delta)
    with pytest.raises(NotImplementedError) as excinfo:
        pt.create_from_ast(constraint_str)
    error_str = ("Error parsing your expression."
         " An operator was used which we do not support: <<")
    assert str(excinfo.value) == error_str

    constraint_str = 'FPR>>4'
    delta = 0.05
    pt = ParseTree(delta)
    with pytest.raises(NotImplementedError) as excinfo:
        pt.create_from_ast(constraint_str)
    error_str = ("Error parsing your expression."
         " An operator was used which we do not support: >>")
    assert str(excinfo.value) == error_str

    constraint_str = 'FPR & FNR'
    delta = 0.05
    pt = ParseTree(delta)
    with pytest.raises(NotImplementedError) as excinfo:
        pt.create_from_ast(constraint_str)
    error_str = ("Error parsing your expression."
         " An operator was used which we do not support: &")
    assert str(excinfo.value) == error_str

    constraint_str = 'FPR//4'
    delta = 0.05
    pt = ParseTree(delta)
    with pytest.raises(NotImplementedError) as excinfo:
        pt.create_from_ast(constraint_str)
    error_str = ("Error parsing your expression."
         " An operator was used which we do not support: //")
    assert str(excinfo.value) == error_str
 
def test_single_conditional_columns_assigned():

    constraint_str = 'abs(Mean_Error|[X]) - 0.1'
    delta = 0.05
    pt = ParseTree(delta)
    pt.create_from_ast(constraint_str)
    assert pt.n_nodes == 4
    assert pt.n_base_nodes == 1
    assert len(pt.base_node_dict) == 1  
    assert pt.root.left.left.conditional_columns == ['X']

def test_multiple_conditional_columns_assigned():

    constraint_str = 'abs(Mean_Error|[X,Y,Z]) - 0.1'
    delta = 0.05
    pt = ParseTree(delta)
    pt.create_from_ast(constraint_str)
    assert pt.n_nodes == 4
    assert pt.n_base_nodes == 1
    assert len(pt.base_node_dict) == 1  
    assert pt.root.left.left.conditional_columns == ['X','Y','Z']

def test_deltas_assigned_equally():
    constraint_str = 'abs((Mean_Error|[M]) - (Mean_Error|[F])) - 0.1'
    delta = 0.05 

    pt = ParseTree(delta)
    pt.create_from_ast(constraint_str)
    pt.assign_deltas(weight_method='equal')
    assert pt.n_nodes == 6
    assert pt.n_base_nodes == 2
    assert len(pt.base_node_dict) == 2  
    assert isinstance(pt.root,InternalNode)
    assert pt.root.name == 'sub'  
    assert pt.root.left.left.left.delta == delta/pt.n_base_nodes
    assert pt.root.left.left.right.delta == delta/pt.n_base_nodes

def test_duplicate_base_nodes():
    constraint_str = 'FPR + 4/FPR - 2.0'
    delta = 0.05 

    pt = ParseTree(delta)
    pt.create_from_ast(constraint_str)
    assert pt.n_base_nodes == 2 
    assert len(pt.base_node_dict) == 1 
    assert pt.base_node_dict['FPR']['computed'] == False
    pt.propagate_bounds(bound_method='random')
    assert pt.base_node_dict['FPR']['computed'] == True

def test_propagate_ttest_bound(generate_data):
    # dummy data for linear regression
    np.random.seed(0)
    numPoints=1000
    from src.model import LinearRegressionModel

    model_instance = LinearRegressionModel()
    X,Y = generate_data(numPoints,loc_X=0.0,loc_Y=0.0,sigma_X=1.0,sigma_Y=1.0)
    rows = np.hstack([np.expand_dims(X,axis=1),np.expand_dims(Y,axis=1)])
    df = pd.DataFrame(rows,columns=['feature1','label'])
    dataset = DataSet(df,meta_information=['feature1','label'],
        regime='supervised',label_column='label')
    
    constraint_str = 'Mean_Squared_Error - 2.0'
    delta = 0.05 

    pt = ParseTree(delta)
    pt.create_from_ast(constraint_str)
    pt.assign_deltas(weight_method='equal')
    assert pt.n_nodes == 3
    assert pt.n_base_nodes == 1
    assert len(pt.base_node_dict) == 1
    assert pt.root.name == 'sub'  
    theta = np.array([0,1])
    pt.propagate_bounds(theta=theta,dataset=dataset,
        model=model_instance,branch='safety_test',bound_method='ttest')
    assert pt.root.lower == pytest.approx(-1.14262257)
    assert pt.root.upper == pytest.approx(-0.98233907)

def test_reset_parse_tree():
    
    constraint_str = '(FPR + FNR) - 0.5'
    delta = 0.05 

    pt = ParseTree(delta)
    pt.create_from_ast(constraint_str)
    pt.assign_deltas(weight_method='equal')
    assert pt.n_base_nodes == 2
    assert len(pt.base_node_dict) == 2
    assert pt.base_node_dict['FPR']['computed'] == False
    assert pt.base_node_dict['FPR']['lower'] == float('-inf')
    assert pt.base_node_dict['FPR']['upper'] == float('inf')
    assert pt.base_node_dict['FNR']['lower'] == float('-inf')
    assert pt.base_node_dict['FNR']['upper'] == float('inf')
    assert pt.base_node_dict['FNR']['computed'] == False

    # propagate bounds
    pt.propagate_bounds(bound_method='random')
    assert len(pt.base_node_dict) == 2
    assert pt.base_node_dict['FPR']['computed'] == True
    assert pt.base_node_dict['FNR']['computed'] == True
    assert pt.base_node_dict['FPR']['lower'] >= 0
    assert pt.base_node_dict['FPR']['upper'] > 0
    assert pt.base_node_dict['FNR']['lower'] >= 0
    assert pt.base_node_dict['FNR']['upper'] > 0

    # reset the node dict 
    pt.reset_base_node_dict()
    assert len(pt.base_node_dict) == 2
    assert pt.base_node_dict['FPR']['computed'] == False
    assert pt.base_node_dict['FNR']['computed'] == False
    assert pt.base_node_dict['FPR']['lower'] == float('-inf')
    assert pt.base_node_dict['FPR']['upper'] == float('inf')
    assert pt.base_node_dict['FNR']['lower'] == float('-inf')
    assert pt.base_node_dict['FNR']['upper'] == float('inf')

def test_single_conditional_columns_propagated():
    np.random.seed(0)
    csv_file = '../datasets/GPA/data_phil_modified.csv'
    columns = ["M","F","SAT_Physics",
           "SAT_Biology","SAT_History",
           "SAT_Second_Language","SAT_Geography",
           "SAT_Literature","SAT_Portuguese_and_Essay",
           "SAT_Math","SAT_Chemistry","GPA"]
           
    loader = DataSetLoader(column_names=columns,
        sensitive_column_names=['M','F'],
        regime='supervised',label_column='GPA')
    dataset = loader.from_csv(csv_file)

    from src.model import LinearRegressionModel
    model_instance = LinearRegressionModel()

    constraint_str = 'abs(Mean_Error|[M]) - 0.1'
    delta = 0.05
    pt = ParseTree(delta)
    pt.create_from_ast(constraint_str)
    pt.assign_deltas(weight_method='equal')

    # propagate the bounds with example theta value
    # theta = np.hstack([np.array([0.0,0.0]),np.random.uniform(-0.05,0.05,10)])
    theta = np.random.uniform(-0.05,0.05,10)
    pt.propagate_bounds(theta=theta,dataset=dataset,
        model=model_instance,branch='safety_test',
        bound_method='ttest')
    assert pt.root.lower == pytest.approx(61.9001779655)
    assert pt.root.upper == pytest.approx(62.1362236720)
    assert len(pt.base_node_dict["(Mean_Error | ['M'])"]['data_dict']['features']) == 22335
    pt.reset_base_node_dict()
    