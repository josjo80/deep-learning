#Thermal solver - Assumes a mass of water, power is held constant as it is delivered to the mass, no phase change
#and no perfusion of blood to the "tissue" (i.e. water mass).

def ThermalSolver(P, T, D, dt, m):
    #Constant definitions
    #C is specific heat capacity
    #C is 4000 J/kgK per "Modeling and numerical simulation of bioheat transfer and biomechanics in soft tissue", Wensheng Shen
    #C is 3600 J/kgC per "Considerations for Thermal Injury Analysis for RF Ablation Devices" Isaac A Chang
    C = 3600
    
    T = (1/(m*C))*((P*dt) + m*C*T) + 1*D #Added in dependancy on D to mimic increase in T as tissue is cooked.
    if T < 100:
        return T
    else:
        return 100