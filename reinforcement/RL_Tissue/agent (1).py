
def Agent(P, n, dt):
    t1 = 100e-3
    t2 = 200e-3
    t3 = 300e-3
    
    if n*dt < t1:
        P = 50
    elif n*dt >= t1 and n*dt < t2:
        P = 5
    elif n*dt >= t2 and n*dt < t3:
        P = 50
    elif n*dt >= t3:
        P = 5
    return P