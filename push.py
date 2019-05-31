# example of python code to compute a particle trajectory
import numpy as np

# set some constants
qe = -1.6022e-19
me = 9.1094e-31
c2 = (2.9979e8)**2
mu0_bar = 2.0e-7

# Set field strengths
V0 = 2.4e6
I0 = 1e5
r_inner = 0.5e-3   # 1 mm dia rod
r_outer = 6.0e-3   # 12 mm dia cathode

# define fields
E0 = V0 / np.log(r_outer/r_inner)
B0 = I0 * mu0_bar
        
def E_function(position):
    r2 = (position[0]**2 + position[1]**2)
    rhat = np.array([position[0], position[1], 0])
    return (E0 / r2) * rhat

def B_function(position):
    r2 = (position[0]**2 + position[1]**2)
    theta_hat = np.array([-position[1], position[0], 0])
    return (B0 / r2) * theta_hat

# set some variables related to the clock
dt = 3e-14
t1 = 3e-12

# define the function that does the boris push
def boris_push(x, p):
    E = E_function(x)
    B = B_function(x)

    vminus = p + dt * E * qe / 2
    m1 = np.sqrt(me**2 + np.dot(p, p)/c2)

    t = dt * B * qe / m1 / 2
    s = 2 * t / (1 + np.dot(t, t))
    vprime = vminus + np.cross(vminus, t)
    vplus = vminus + np.cross(vprime, s)
    p = vplus + dt * E * qe / 2
    m2 = np.sqrt(me**2 + np.dot(p, p)/c2)
    x = x + dt * p / m2
    return x, p

# set initial conditions
x = np.array([0.05/100,0,0])
p = np.array([2.345e6, 0, 0]) * 1.6022e-19 / 2.9979e8

# Do initial half push
m2 = np.sqrt(me**2 + np.dot(p, p)/c2)
x = x + (dt/2) * p / m2

# Do the updates
t = 0
while t < t1 and (x[0]**2 + x[1]**2) > (0.05/100)**2:
    t = t + dt
    x, p = boris_push(x, p)
    print("time:", t, "x:", x, "p:", p)

print("Simulation finished")
print("Final values:")
print("time:", t, "x:", x, "p:", p)


