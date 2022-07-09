def define_hyperparameter_and_setting_dict():
    the_dict = {}

    # the_dict["env"] = "mountaincar"
    the_dict["env"] = "n_step_mountaincar"

    the_dict["agent"] = "mountain_car_rough_solution"

    the_dict["num_episodes"] = 3
    the_dict["num_trials"] = 2
    the_dict["vis"] = False
    return the_dict