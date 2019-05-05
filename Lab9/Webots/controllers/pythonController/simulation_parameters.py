"""Simulation parameters"""


class SimulationParameters(dict):
    """Simulation parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, **kwargs):
        super(SimulationParameters, self).__init__()
        # Default parameters
        self.n_body_joints = 10
        self.n_legs_joints = 4
        self.simulation_duration = 30
        self.phase_lag = 0
        self.amplitude_gradient = [0,0.5] 
        # Feel free to add more parameters (ex: MLR drive)
        # self.drive_mlr = ...
        # ...
        self.freqs = 1
        self.couplingBody = 10
        self.couplingLeg = 30
        self.rate = 1
        # Update object with provided keyword arguments
        self.update(kwargs)  # NOTE: This overrides the previous declarations
        

