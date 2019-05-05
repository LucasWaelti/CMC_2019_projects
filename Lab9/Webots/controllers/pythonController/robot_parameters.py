"""Robot parameters"""

import numpy as np
import cmc_pylog as pylog


class RobotParameters(dict):
    """Robot parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, parameters):
        super(RobotParameters, self).__init__()

        # Initialise parameters
        self.n_body_joints = parameters.n_body_joints # 10
        self.n_legs_joints = parameters.n_legs_joints # 4
        self.n_joints = self.n_body_joints + self.n_legs_joints # 14
        self.n_oscillators_body = 2*self.n_body_joints # 20
        self.n_oscillators_legs = self.n_legs_joints # 4
        self.n_oscillators = self.n_oscillators_body + self.n_oscillators_legs # 24
        self.freqs = np.zeros(self.n_oscillators) # 24
        self.coupling_weights = np.zeros([ # 24 x 24
            self.n_oscillators,
            self.n_oscillators
        ])
        self.phase_bias = np.zeros([self.n_oscillators, self.n_oscillators]) # 24 x 24
        self.rates = np.zeros(self.n_oscillators) # 24 
        self.nominal_amplitudes = np.zeros(self.n_oscillators) # 24 
        self.update(parameters)

    def update(self, parameters):
        """Update network from parameters"""
        self.set_frequencies(parameters)  # f_i
        self.set_coupling_weights(parameters)  # w_ij
        self.set_phase_bias(parameters)  # theta_i
        self.set_amplitudes_rate(parameters)  # a_i
        self.set_nominal_amplitudes(parameters)  # R_i

    def set_frequencies(self, parameters):
        """Set frequencies"""
        #pylog.warning("Frequencies weights must be set")
        self.freqs[:self.n_oscillators] = np.ones(self.n_oscillators) * parameters.freqs
        #print(self.freqs)

    def set_coupling_weights(self, parameters):
        """Set coupling weights - w_ij is coupling from j to i"""
        # Reference table to identify the spine oscillators influenced by each leg
        spine_index = [0,int(self.n_oscillators_body/2),
                        int(self.n_oscillators_body/4),
                        int(3*self.n_oscillators_body/4)]

        for i in range(0, self.n_oscillators):
            for j in range(0, self.n_oscillators): 
                if i < self.n_oscillators_body and i < self.n_oscillators_body:
                    # Compute the couplings within the spine only
                    if (i == j + 1) or (i == j - 1) or (i == j + 10) or (i == j - 10):
                        self.coupling_weights[i][j] = parameters.couplingBody
                else:
                    # Compute the couplings between the legs and the spine 
                    if i >= self.n_oscillators_body and j >= self.n_oscillators_body:
                        # both i and j are legs - same weight as body coupling 
                        # Only indluenced by other legs, except the one on the diagonal
                        legi = i - self.n_oscillators_body 
                        legj = j - self.n_oscillators_body # express the legs between 0 and 3

                        if (legi == 0 or legi == 3) and (legj == 1 or legj == 2):
                            self.coupling_weights[i][j] = parameters.couplingBody 
                        elif (legi == 1 or legi == 2) and (legj == 0 or legj == 3):
                            self.coupling_weights[i][j] = parameters.couplingBody 
                    
                    # Case where only i represents a leg has no effect
                     
                    elif i < self.n_oscillators_body and j >= self.n_oscillators_body:
                        # only j represents a leg, i is a spine oscillator 
                        legj = j - self.n_oscillators_body
                        if i in range(spine_index[legj], spine_index[legj] + self.n_oscillators_body/4):
                            self.coupling_weights[i][j] = parameters.couplingLeg 
        
        #pylog.warning("Coupling weights must be set")

    def set_phase_bias(self, parameters):
        """Set phase bias"""
        for i in range(0, self.n_oscillators_body):
            for j in range(0, self.n_oscillators_body):
                if i >= self.n_oscillators_body and j >= self.n_oscillators_body: 
                    # Compute inter leg phases 
                    legi = i - self.n_oscillators_body 
                    legj = j - self.n_oscillators_body # express the legs between 0 and 3

                    if (legi == 0 or legi == 3) and (legj == 1 or legj == 2):
                        self.phase_bias[i][j] = np.pi 
                    elif (legi == 1 or legi == 2) and (legj == 0 or legj == 3):
                        self.phase_bias[i][j] = np.pi  
                else:
                    # Compute phases within the spine 
                    if (i == j + 1):
                        self.phase_bias[i][j] = parameters.phase_lag
                    if (i == j - 1):
                        self.phase_bias[i][j] = -parameters.phase_lag
                    if (i == j + 10):
                        self.phase_bias[i][j] = np.pi
                    if (i == j - 10):
                        self.phase_bias[i][j] = -np.pi
        #pylog.warning("Phase bias must be set")

    def set_amplitudes_rate(self, parameters):
        """Set amplitude rates - a"""
        self.rates[:self.n_oscillators] = np.ones(self.n_oscillators) * parameters.rate
        #pylog.warning("Convergence rates must be set")

    def set_nominal_amplitudes(self, parameters):
        """Set nominal amplitudes"""
        slope = parameters.amplitude_gradient[0]
        offset = parameters.amplitude_gradient[1]
        arr = np.array([slope * i + offset for i in range(0, self.n_body_joints)])
        self.nominal_amplitudes[:self.n_oscillators_body] = np.hstack((arr, arr))
        # Constant amplitude of legs oscillations 
        self.nominal_amplitudes[self.n_oscillators_body:self.n_oscillators] = np.ones(self.n_legs_joints) 
        #pylog.warning("Nominal amplitudes must be set")

