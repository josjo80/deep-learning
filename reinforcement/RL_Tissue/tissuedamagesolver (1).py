#TissueDamageSolver - Integrates the damage to tissue over time for a given temperature.  Assumes Dold is managed in
#main loop.
import math

def TissueDamageSolver(T, D, dt):
    #n is the step of the activation.  Assumes dt timestep.
    A = 5.6e63 #5.6e63 (old 7.39e39) 1/s for artery per "Rate process model for arterial tissue thermal damage: implications on vessel photocoagulation." by Agah R, Pearce JA, Welch AJ
    dE = 4.3e5 #4.3e5 (old 2.577e5) J/mol for artery per "Rate process model for arterial tissue thermal damage: implications on vessel photocoagulation." by Agah R, Pearce JA, Welch AJ
    R = 8.3144598 #Universal gas constant, J/K*mol
    T = T + 273.15
    D = D + (A*math.exp(-dE/(R*T))*dt)
    return D