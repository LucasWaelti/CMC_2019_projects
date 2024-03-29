"""Oscillator network ODE"""

import numpy as np

from solvers import euler, rk4
from robot_parameters import RobotParameters

limb_spine_lag = 0 # HACK - dirty, dirty... 

def network_ode(_time, state, parameters):
    """Network_ODE

    returns derivative of state (phases and amplitudes)

    """
    phases = state[:parameters.n_oscillators]
    amplitudes = state[parameters.n_oscillators:2*parameters.n_oscillators]
    
    # Compute the state 
    dphases = np.zeros_like(phases)
    for i in range(0, parameters.n_oscillators):
        dphases[i] += 2 * np.pi * parameters.freqs[i]
        for j in range(0, parameters.n_oscillators):
            # Inter spline coupling
            #if (i == j + 1) or (i == j - 1) or (i == j + 10) or (i == j - 10):
            dphases[i] += amplitudes[j] * parameters.coupling_weights[i][j] * \
                np.sin(phases[j] - phases[i] - parameters.phase_bias[i][j])
    
    damplitudes = np.zeros_like(amplitudes)
    for i in range(0, parameters.n_oscillators):
        damplitudes[i] = parameters.rates[i] * (parameters.nominal_amplitudes[i] - amplitudes[i])

    return np.concatenate([dphases, damplitudes])

def motor_output(phases, amplitudes):
    """Motor output - q_i"""
    global limb_spine_lag

    n_body_joints = int(len(amplitudes) / 2 - 2) # 10 
    motor_output = np.zeros(int(len(amplitudes) / 2 + 2)) # 14

    # For the spine
    for i in range(0, n_body_joints):
        motor_output[i] = amplitudes[i] * (1 + np.cos(phases[i])) \
            - amplitudes[i + n_body_joints] * ( 1 + np.cos(phases[i + n_body_joints]))
    
    # For the legs 
    for i in range(n_body_joints, n_body_joints + 4):
        if (amplitudes[10+i] > 0.001):
            # TODO - avoid hardcoding this! Should not require explicit offset like this!! Bad! 
            if i == 11 or i == 12:
                motor_output[i] = -phases[i]-limb_spine_lag + np.pi
                #motor_output[i] = amplitudes[10+i]*(1 + np.cos(-phases[i]-limb_spine_lag + np.pi)) # NOTE - comment this when done debugging 
            else:
                motor_output[i] = -phases[i]-limb_spine_lag 
                #motor_output[i] = amplitudes[10+i]*(1 + np.cos(-phases[i]-limb_spine_lag)) # NOTE - comment this when done debugging 
        else:
            # Avoid leg rewind 
            """ i = 1
            while True:
                diff = np.abs(phases[i]) - i*2*np.pi 
                if diff <= 2*np.pi:
                    motor_output[i] = -(phases[i] + diff if phases[i] <=0 else phases[i] - diff)
                    break 
                i+=1 """ 
            motor_output[i] = 0
    return motor_output


class ODESolver(object):
    """ODE solver with step integration"""

    def __init__(self, ode, timestep, solver=rk4):
        super(ODESolver, self).__init__()
        self.ode = ode
        self.solver = solver
        self.timestep = timestep
        self._time = 0

    def integrate(self, state, *parameters):
        """Step"""
        diff_state = self.solver(
            self.ode,
            self.timestep,
            self._time,
            state,
            *parameters
        )
        self._time += self.timestep
        return diff_state

    def time(self):
        """Time"""
        return self._time


class RobotState(np.ndarray):
    """Robot state"""

    def __init__(self, *_0, **_1):
        super(RobotState, self).__init__()
        self[:] = 0.0

    @classmethod
    def salamandra_robotica_2(cls):
        """State of Salamandra robotica 2"""
        return cls(2*24, dtype=np.float64, buffer=np.zeros(2*24))

    @property
    def phases(self):
        """Oscillator phases"""
        return self[:24]

    @phases.setter
    def phases(self, value):
        self[:24] = value

    @property
    def amplitudes(self):
        """Oscillator phases"""
        return self[24:]

    @amplitudes.setter
    def amplitudes(self, value):
        self[24:] = value


class SalamanderNetwork(ODESolver):
    """Salamander oscillator network"""

    def __init__(self, timestep, parameters):
        global limb_spine_lag 
        super(SalamanderNetwork, self).__init__(
            ode=network_ode,
            timestep=timestep,
            solver=rk4  # Feel free to switch between Euler (euler) or
                        # Runge-Kutta (rk4) integration methods
        )
        # Store the phase lag between legs and spine - HACK 
        limb_spine_lag = parameters.limb_spine_phase_lag
        # States 
        self.state = RobotState.salamandra_robotica_2()
        # Parameters
        self.parameters = RobotParameters(parameters)
        # Set initial state
        self.state.phases = 1e-4*np.random.ranf(self.parameters.n_oscillators)

    def step(self):
        """Step"""
        self.state += self.integrate(self.state, self.parameters)
        return self.state 

    def get_motor_position_output(self):
        """Get motor position"""
        return motor_output(self.state.phases, self.state.amplitudes)

