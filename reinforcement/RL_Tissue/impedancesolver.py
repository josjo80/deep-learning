#ImpedanceSolver - Assumes saline solution that matches tissue conductance.  Takes in tissue temp, solution normalization,
#and height/thickness of tissue.  Returns conductance, sig, and impedance, Z based on temperature.

def ImpedanceSolver(D, T, N, h):
    #sigma is based on Eq 6 from "Considerations for Thermal Injury Analysis for RF Ablation Devices" Isaac A Chang
    delta = 25.0 - T
    sig25N = N*(10.394 - 2.3776*N + 0.68258*N**2 - 9.13538*N**3 + 1.0086e-2*N**4)
    sig = sig25N*(1.0 - 1.962e-2*delta + 8.08e-5*delta**2 - N*delta*(3.020e-5 + 3.922e-5*delta + N*(1.721e-5 - 6.584e-6*delta)))
    Z = 1/(sig*h) + (10*D)**2 #Adding dependance on D damage, to mimic the increase in Z as tissue is cooked.
    
    return sig, Z