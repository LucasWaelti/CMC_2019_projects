""" Lab 5 - Exercise 1 """

import matplotlib.pyplot as plt
import numpy as np

import cmc_pylog as pylog
from muscle import Muscle
from mass import Mass
from cmcpack import DEFAULT, parse_args
from cmcpack.plot import save_figure
from system_parameters import MuscleParameters, MassParameters
from isometric_muscle_system import IsometricMuscleSystem
from isotonic_muscle_system import IsotonicMuscleSystem

DEFAULT["label"] = [r"$\theta$ [rad]", r"$d\theta/dt$ [rad/s]"]

# Global settings for plotting
# You may change as per your requirement
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=14.0)     # fontsize of the axes title
plt.rc('axes', labelsize=14.0)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)    # fontsize of the tick labels

DEFAULT["save_figures"] = True


def exercise1a():
    """ Exercise 1a
    The goal of this exercise is to understand the relationship
    between muscle length and tension.
    Here you will re-create the isometric muscle contraction experiment.
    To do so, you will have to keep the muscle at a constant length and
    observe the force while stimulating the muscle at a constant activation."""

    # Defination of muscles
    parameters = MuscleParameters()
    pylog.warning("Loading default muscle parameters")
    pylog.info(parameters.showParameters())
    pylog.info("Use the parameters object to change the muscle parameters")

    # Create muscle object
    muscle = Muscle(parameters)

    pylog.warning("Isometric muscle contraction to be completed")

    # Instatiate isometric muscle system
    sys = IsometricMuscleSystem()

    # Add the muscle to the system
    sys.add_muscle(muscle)

    # You can still access the muscle inside the system by doing
    # >>> sys.muscle.L_OPT # To get the muscle optimal length

    # Evalute for a single muscle stretch
    muscle_stretch = 0.2

    # Evalute for a single muscle stimulation
    muscle_stimulation = 1.

    # Set the initial condition
    x0 = [0.0, sys.muscle.L_OPT]
    # x0[0] --> muscle stimulation intial value
    # x0[1] --> muscle contracticle length initial value

    # Set the time for integration
    t_start = 0.0
    t_stop = 0.2
    time_step = 0.001

    time = np.arange(t_start, t_stop, time_step)

    # Run the integration
    result = sys.integrate(x0=x0,
                           time=time,
                           time_step=time_step,
                           stimulation=muscle_stimulation,
                           muscle_length=muscle_stretch)

    # Plotting
    plt.figure('Isometric muscle experiment')
    plt.plot(result.time, result.tendon_force,label='tendon force')
    plt.plot(result.time, result.passive_force,label='passive force')
    plt.plot(result.time, result.active_force,label='active force')
    plt.plot(result.time, result.l_ce,label='l_ce')
    plt.legend()
    plt.title('Isometric muscle experiment')
    plt.xlabel('Time [s]')
    plt.ylabel('Muscle Force')
    plt.grid()
    
    # Run 1.a and 1.b - relation between l_ce and active/oassive forces 
    ms=np.arange(start=0.0,stop=0.32,step=0.005)
    plotRelationLceAPForces(muscle_stretch=ms,muscle_stimulation=0.1)
    plotRelationLceAPForces(muscle_stretch=ms,muscle_stimulation=0.5)
    plotRelationLceAPForces(muscle_stretch=ms,muscle_stimulation=1.)
    
    # Run 1.c 
    plotRelationLceAPForces(ms,l_opt=0.09)
    plotRelationLceAPForces(ms,l_opt=0.13)

def plotRelationLceAPForces(muscle_stretch,muscle_stimulation=1.,l_opt=0.11):
    # Defination of muscles
    parameters = MuscleParameters()
    # Create muscle object
    muscle = Muscle(parameters)    
    # Instatiate isometric muscle system
    sys = IsometricMuscleSystem()
    # Add the muscle to the system
    sys.add_muscle(muscle)
    sys.muscle.L_OPT = l_opt
    # Set the initial condition
    x0 = [0.0, sys.muscle.L_OPT] # x0[0] --> muscle stimulation intial value 
                                 # x0[1] --> muscle contracticle length initial value
    # Set the time for integration
    t_start = 0.0
    t_stop = 0.2
    time_step = 0.001
    time = np.arange(t_start, t_stop, time_step)
    
    # Store the results 
    l_ce = []
    active = []
    passive = []
    tendon = []
    # Run the experiment for different length of the MTU
    for l in muscle_stretch:
        # Run the integration
        result = sys.integrate(x0=x0,
                           time=time,
                           time_step=time_step,
                           stimulation=muscle_stimulation,
                           muscle_length=l)
        l_ce.append(result.l_ce[-1])
        active.append(result.active_force[-1])
        passive.append(result.passive_force[-1])
        tendon.append(result.tendon_force[-1])
        
    plt.figure('Active/passive forces as function of l_ce\n'+
               '(activation: {}, l_opt: {})'
               .format(muscle_stimulation,l_opt))
    plt.plot(l_ce, active,label='active force')
    plt.plot(l_ce, passive,label='passive force')
    plt.plot(l_ce, tendon,label='tendon force')
    plt.legend()
    plt.title('Isometric muscle experiment\nActive/passive forces as function '+  
              'of l_ce\n(activation: {}, l_opt: {})'.format(muscle_stimulation,l_opt))
    plt.xlabel('l_ce [m]')
    plt.ylabel('Force [N]')
    axes = plt.gca()
    axes.set_xlim([0.05,0.2])
    axes.set_ylim([0,1700])
    plt.grid()
    


def exercise1d():
    """ Exercise 1d

    Under isotonic conditions external load is kept constant.
    A constant stimulation is applied and then suddenly the muscle
    is allowed contract. The instantaneous velocity at which the muscle
    contracts is of our interest."""

    # Defination of muscles
    muscle_parameters = MuscleParameters()
    print(muscle_parameters.showParameters())

    mass_parameters = MassParameters()
    print(mass_parameters.showParameters())

    # Create muscle object
    muscle = Muscle(muscle_parameters)

    # Create mass object
    mass = Mass(mass_parameters)

    pylog.warning("Isotonic muscle contraction to be implemented")

    # Instatiate isotonic muscle system
    sys = IsotonicMuscleSystem()

    # Add the muscle to the system
    sys.add_muscle(muscle)

    # Add the mass to the system
    sys.add_mass(mass)

    # You can still access the muscle inside the system by doing
    # >>> sys.muscle.L_OPT # To get the muscle optimal length

    # Evalute for a single load
    load = 100.

    # Evalute for a single muscle stimulation
    muscle_stimulation = 1.

    # Set the initial condition
    x0 = [0.0, sys.muscle.L_OPT,
          sys.muscle.L_OPT + sys.muscle.L_SLACK, 0.0]
    # x0[0] - -> activation
    # x0[1] - -> contractile length(l_ce)
    # x0[2] - -> position of the mass/load
    # x0[3] - -> velocity of the mass/load

    # Set the time for integration
    t_start = 0.0
    t_stop = 0.3
    time_step = 0.001
    time_stabilize = 0.2

    time = np.arange(t_start, t_stop, time_step)

    # Run the integration
    result = sys.integrate(x0=x0,
                           time=time,
                           time_step=time_step,
                           time_stabilize=time_stabilize,
                           stimulation=muscle_stimulation,
                           load=load)

    # Plotting
    plt.figure('Isotonic muscle experiment')
    plt.plot(result.time, result.v_ce)
    plt.title('Isotonic muscle experiment')
    plt.xlabel('Time [s]')
    plt.ylabel('Muscle contractilve velocity')
    plt.grid()
    
    # Run 1.d
    load = np.arange(0,1000,20)
    plotVceLoad(load,[0.1,0.5,1])
    
def plotVceLoad(load,ms=[1.]):
    # Defination of muscles
    muscle_parameters = MuscleParameters()
    mass_parameters = MassParameters()
    # Create muscle object
    muscle = Muscle(muscle_parameters)
    # Create mass object
    mass = Mass(mass_parameters)
    # Instatiate isotonic muscle system
    sys = IsotonicMuscleSystem()
    # Add the muscle to the system
    sys.add_muscle(muscle)
    # Add the mass to the system
    sys.add_mass(mass)
    
    # Evalute for a single load
    #load = 100.

    # Set the initial condition
    x0 = [0.0, sys.muscle.L_OPT,
          sys.muscle.L_OPT + sys.muscle.L_SLACK, 0.0]
    # x0[0] - -> activation
    # x0[1] - -> contractile length(l_ce)
    # x0[2] - -> position of the mass/load
    # x0[3] - -> velocity of the mass/load

    # Set the time for integration
    t_start = 0.0
    t_stop = 0.3
    time_step = 0.001
    time_stabilize = 0.2

    time = np.arange(t_start, t_stop, time_step)
    
    plt.figure('Max Velocity-Tension curve')
    for s in ms:
        muscle_stimulation = s
        v = []
        for l in load:
            # Run the integration
            result = sys.integrate(x0=x0,
                                   time=time,
                                   time_step=time_step,
                                   time_stabilize=time_stabilize,
                                   stimulation=muscle_stimulation,
                                   load=l)
            # Find the max or min speed achieved
            i = np.argmax(np.abs(result.v_ce))
            v.append(-result.v_ce[i])
            #if result[i].l_mtc < sys.muscle.L_OPT + sys.muscle.L_SLACK:
        
        for i in range(len(v)):
            if i >= 1 and v[i]*v[i-1] <=0:
                plt.plot(load[i],v[i],color='green', marker='x', linestyle='dashed',
                         linewidth=2, markersize=12)
        plt.plot(load, v,label='maximal speed\nMuscle stimulation: {}'.format(s))
        plt.legend()
        plt.title('Isotonic muscle experiment\nMax Velocity-Tension curve')
        plt.xlabel('load [kg]')
        plt.ylabel('CE speed [m/s]') 
        #axes = plt.gca()
        #axes.set_xlim([0.05,0.2])
        #axes.set_ylim([0,1700])
    plt.grid()
    

def exercise1():
    exercise1a()
    exercise1d()

    if DEFAULT["save_figures"] is False:
        plt.show()
    else:
        figures = plt.get_figlabels()
        print(figures)
        pylog.debug("Saving figures:\n{}".format(figures))
        for fig in figures:
            plt.figure(fig)
            save_figure(fig)
            plt.close(fig)


if __name__ == '__main__':
    from cmcpack import parse_args
    parse_args()
    exercise1()

