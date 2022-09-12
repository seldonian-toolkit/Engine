def define_hyperparameter_and_setting_dict():
    the_dict = {}

    # the_dict["env"] = "gridworld"
    # the_dict["env"] = "mountaincar"
    # the_dict["env"] = "n_step_mountaincar"
    the_dict["env"] = "simglucose"

    # the_dict["agent"] = "mountain_car_rough_solution"
    # the_dict["agent"] = "discrete_random"
    the_dict["agent"] = "Parameterized_non_learning_softmax_agent"
    # the_dict["agent"] = "Keyboard_gridworld"

    the_dict["basis"] = "Fourier"
    the_dict["order"] = 2
    the_dict["max_coupled_vars"] = -1

    the_dict["num_episodes"] = 1
    the_dict["vis"] = False
    return the_dict
