"""Oscillator network ODE"""

import numpy as np
import cmc_pylog as pylog

from solvers import euler, rk4


def phases_ode(time, phases, freqs, coupling_weights, phases_desired, amplitudes):
    """Network phases ODE"""
    '''print(phases_desired.shape)(20, 20) [i,j] represents relation between i and j
    print(phases.shape)(20,) **state**
    print(freqs.shape)(20,)
    print(coupling_weights.shape)(20, 20)'''

    # Oscillator amplitude r (missing in provided implementation) -> assumed to be 1
    r = amplitudes#np.ones(20)

    # State derivative (phases derivative)
    dphases = np.zeros_like(phases)

    """ Aurélien's version: 
    for i in range(len(freqs)):
        dphases[i] += 2 * np.pi * freqs[i]
        for j in range(0, len(freqs)):
            dphases[i] += 1 * coupling_weights[i][j] * np.sin(phases[j] - phases[i] - phases_desired[i][j]) """

    for i in range(len(freqs)):
        dphases[i] += 2 * np.pi * freqs[i]
        for j in range(len(freqs)):
            if i == j + 1 or  i == j - 1 or i == j + 10 or i == j - 10:
                dphases[i] += r[j] * coupling_weights[i,j] * np.sin(phases[j] - phases[i] - phases_desired[i,j]) 
                      
    return dphases


def amplitudes_ode(time, amplitudes, rate, amplitudes_desired):
    """Network amplitudes ODE"""
    damplitudes = np.zeros_like(amplitudes)
    for i in range(0, len(amplitudes)):
        damplitudes[i] = rate[i] * (amplitudes_desired[i] - amplitudes[i])
    return damplitudes

def motor_output(phases_left, phases_right, amplitudes_left, amplitudes_right):
    """Motor output"""
    dmotor_output = np.zeros_like(amplitudes_left)
    for i in range(0, len(dmotor_output)):
        dmotor_output[i] = amplitudes_left[i] * (1 + np.cos(phases_left[i])) - amplitudes_right[i] * (1 + np.cos(phases_right[i]))
    
    return dmotor_output


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
        dstate = self.solver(
            self.ode,
            self.timestep,
            self._time,
            state,
            *parameters
        )
        self._time += self.timestep
        return dstate

    def time(self):
        """Time"""
        return self._time


class PhaseEquation(ODESolver):
    """Phase ODE equation"""

    def __init__(self, timestep, freqs, phase_lag):
        super(PhaseEquation, self).__init__(phases_ode, timestep, euler)
        self.n_joints = 10
        self.phases = 1e-4*np.random.ranf(2*self.n_joints) # shape (20,)
        self.freqs = np.zeros(2*self.n_joints)
        self.coupling_weights = np.zeros([2*self.n_joints, 2*self.n_joints])
        self.phases_desired = np.zeros([2*self.n_joints, 2*self.n_joints])     #size initialization
        self.set_parameters(freqs, phase_lag)

    def set_parameters(self, freqs, phase_lag):
        """Set parameters of the network"""

        # Set coupling weights
        # Set desired phases
        # Set frequencies
        
        for i in range(0, len(self.freqs)):
            self.freqs[i] = freqs[i] # modified freqs to be an array
            for j in range(0, len(self.freqs)):
                if i == j + 1:
                    self.coupling_weights[i][j] = 10
                    self.phases_desired[i][j] = phase_lag
                if i == j - 1:
                    self.coupling_weights[i][j] = 10
                    self.phases_desired[i][j] = -phase_lag
                if i == j + 10:
                    self.coupling_weights[i][j] = 10
                    self.phases_desired[i][j] = np.pi #phase_lag
                if i == j - 10:
                    self.coupling_weights[i][j] = 10
                    self.phases_desired[i][j] = -np.pi #phase_lag

    def step(self,amplitudes):
        """Step"""
        self.phases += self.integrate(
            self.phases, # phases are the state
            self.freqs,  # All remaining arguments are combined into *parameters
            self.coupling_weights,
            self.phases_desired,
            amplitudes 
        )


class AmplitudeEquation(ODESolver):
    """Amplitude ODE equation"""

    def __init__(self, timestep, amplitudes, turn):
        super(AmplitudeEquation, self).__init__(amplitudes_ode, timestep, euler)
        self.n_joints = 10
        self.amplitudes = np.zeros(2*self.n_joints)
        self.rates = np.zeros(2*self.n_joints)
        self.amplitudes_desired = np.zeros(2*self.n_joints)
        self.set_parameters(amplitudes, turn)

    def set_parameters(self, amplitudes, turn):
        """Set parameters of the network"""

        # Set convergence rates ???????????
        # Set desired amplitudes

        # Amplitude contains now [Rhead,Rtail] - interpolate from 0 to 10 between those 2 values
        slope = (amplitudes[1]-amplitudes[0])/9
        offset = amplitudes[0]
        
        for i in range(0, len(self.rates)):
            self.rates[i] = 1
            if i >= 10:
                self.amplitudes_desired[i] = slope*(i-10) + offset + turn #amplitudes[i]
            else: 
                self.amplitudes_desired[i] = slope*i + offset - turn 
        

    def step(self):
        """Step"""
        self.amplitudes += self.integrate(
            self.amplitudes,
            self.rates,
            self.amplitudes_desired
        )
        return self.amplitudes


class SalamanderNetwork(object):
    """Salamander oscillator network"""

    def __init__(self, timestep, freqs, amplitudes, phase_lag, turn):
        super(SalamanderNetwork, self).__init__()
        # Phases
        self.phase_equation = PhaseEquation(
            timestep,
            freqs,
            phase_lag
        )
        # Amplitude
        self.amplitude_equation = AmplitudeEquation(
            timestep,
            amplitudes,
            turn
        )

    def step(self):
        """Step"""
        # Correction to take amplitude into account in phase ode!!
        amplitudes = self.amplitude_equation.step()
        self.phase_equation.step(amplitudes) 

    def get_motor_position_output(self):
        """Get motor position"""
        return motor_output(
            self.phase_equation.phases[:10],
            self.phase_equation.phases[10:],
            self.amplitude_equation.amplitudes[:10],
            self.amplitude_equation.amplitudes[10:]
        )

