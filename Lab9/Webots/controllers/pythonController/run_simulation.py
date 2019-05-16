"""Run simulation"""

import cmc_pylog as pylog
from cmc_robot import SalamanderCMC

def run_simulation(world, parameters, timestep, n_iterations, logs, transition=False):
    """Run simulation"""

    # Set parameters
    pylog.info(
        "Running new simulation:\n  {}".format("\n  ".join([
            "{}: {}".format(key, value)
            for key, value in parameters.items()
        ]))
    )

    # Setup salamander control
    salamander = SalamanderCMC(
        world,
        n_iterations,
        logs=logs,
        parameters=parameters
    )
    network = salamander.network 

    # Simulation
    x_threshold = 1.2 
    iteration = 0
    while world.step(timestep) != -1:
        iteration += 1
        if iteration >= n_iterations:
            break
        
        #print(salamander.gps.getValues())
        if transition:
            if x_threshold < salamander.gps.getValues()[0]:
                # Switch to swimming 
                #print('Now swimming')
                parameters.drive = 5 
                salamander.network.parameters.update(parameters)
                
        salamander.step()
        
    # Log data
    pylog.info("Logging simulation data to {}".format(logs))
    salamander.log.save_data()

'''

# Logs - copied from run_network 
    phases_log = np.zeros([
        n_iterations,
        len(network.state.phases)
    ])
    phases_log[0, :] = network.state.phases
    amplitudes_log = np.zeros([
        n_iterations,
        len(network.state.amplitudes)
    ])
    amplitudes_log[0, :] = network.state.amplitudes
    freqs_log = np.zeros([
        n_iterations,
        len(network.parameters.freqs)
    ])
    freqs_log[0, :] = network.parameters.freqs
    outputs_log = np.zeros([
        n_iterations,
        len(network.get_motor_position_output())
    ])
    outputs_log[0, :] = network.get_motor_position_output()


# Copied from run_network 
        phases_log[i+1, :] = network.state.phases
        amplitudes_log[i+1, :] = network.state.amplitudes
        outputs_log[i+1, :] = network.get_motor_position_output()
        freqs_log[i+1, :] = network.parameters.freqs
    '''
