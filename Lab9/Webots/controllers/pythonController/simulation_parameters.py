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
        self.amplitude = 0.05
        # Feel free to add more parameters (ex: MLR drive)
        # self.drive_mlr = ...
        # ...
        self.couplingBody = 10
        self.couplingLeg = 30
        self.rate = 20
        self.dBody = [1, 5]
        self.dLimb = [1, 3]
        self.cVBody = [0.2, 0.3]
        self.cVLimb = [0.2, 0.0]        # [0.2, 0.0] originally 
        self.cRBody = [0.0065, 0.05]    # [0.065, 0.196] originally 
        self.cRLimb = [0.131, 0.131]
        self.vSat = 0
        self.RSat = 0
        self.turn = 0 # between -1 (left) and 1 (right)
        self.drive = 0
        # Update object with provided keyword arguments
        self.update(kwargs)  # NOTE: This overrides the previous declarations
        

