

### env/simglucose_gym_env
    
    - Just an openAI gym api wrapper for env.
    - Choose patient id
    - load patient's body parameters and other medical equipments
    - Customize reward function
    - Act using basal but no bolus. (Can be changed) 
    - calls simulation/env

### simulation/env

    - This is the main 'environment' file.
    - Run multiple minutes for a patient
    - Record all the measurements
    - Stopping criteria for the simulation trial
    - calls patient/t1dpatient

### patient/t1dpatient

    - This file contains one step dynamics of the 'environment'.
    - Run one minute of patient
    - load patient specific features
    - solve ODE
    
#### Other Important files:
    - analysis/risk: reward function
    - controller/basal_bolus_ctrller: standard baseline controller
    - simulation/scenario_gen: Food timings