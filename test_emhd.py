from turbopy import Simulation, Module
import numpy as np
from scipy import sparse
from EMHD import EMHD, PlasmaChemistry, ThermalFluidPlasma, FluidElectrons
from pathlib import Path
import scipy.special as special      # for i1, k1 modified Bessel functions
import scipy.integrate as integrate

import matplotlib.pyplot as plt


p = Path('chemistry/N2_Rates_TT.txt')
RateFileList=[p]
initial_conditions={      'e':{'pressure':1e-5*1.0/760,'velocity':1e-6,'energy':0.1},
                     'N2(X1)':{'pressure':1.0/760,'velocity':0.0,'energy':1.0/40.0}
                     }                    


sim_config = {"Modules": [
        {"name": "PlasmaChemistry",
            "RateFileList": RateFileList
        },
        {"name": "ThermalFluidPlasma",
            "initial_conditions": initial_conditions
        },
        {"name": "FluidElectrons",
            "initial_conditions": initial_conditions
        },
        {"name": "EMHD"}
    ],
    "Diagnostics": [
    ],
    "Tools": [],
    "Grid": {"N": 50,
             "r_min": 0, "r_max": 0.1,
            },
    "Clock": {"start_time": 0,
              "end_time":  1, 
              "num_steps": 2,

              }
    }
    
sim = Simulation(sim_config)
sim.prepare_simulation()

r = sim.grid.r
J = sim.grid.generate_field(1)
beam_radius = 0.05
peak_current = 1
J[:] = -2 * (r/beam_radius**2) * peak_current * np.exp(-(r/beam_radius)**2)

mod = sim.modules[-1]
ddr = mod.rad_diff

def plotem(title):
    fig = plt.figure()
    fig.suptitle(title)
#    plt.subplot(4,1,1)
    plt.plot(r, mod.B, '.-')
    plt.title("B field")
    plt.show()
#    plt.subplot(4,1,2)
    plt.plot(r, mod.Omega, '.-')
    plt.title("Omega")
    plt.show()
#    plt.subplot(4,1,3)
    plt.plot(r, mod.J_plasma, '.-')
    plt.title("J plasma")
    plt.show()
#    plt.subplot(4,1,4)
    plt.plot(r, mod.dJdr, '.-')
    plt.title("dJdr beam")
    plt.show()

mod.dJdr[:] = J

plotem("before")

mod.update()

plotem("after")

##
# B = I1 * int_R(K1 * source) + K1 * int_L(I1 * source)

# alpha changes because density can change? Is this ok?
i1_grid = special.i1(r * mod.alpha)
k1_grid = special.k1(r * mod.alpha)
# k1_grid[0] = 0  # artificially remove singularity at origin

source = mod.Omega + mod.mu0 * mod.dJdr
I1 = i1_grid * integrate.cumtrapz((k1_grid * mod.source)[::-1], dx=sim.grid.dr, initial=0)[::-1]
K1 = k1_grid * integrate.cumtrapz((i1_grid * mod.source), dx=sim.grid.dr, initial=0)

# self.B[:] = I1 + K1
# self.B -= self.B[0]

plt.plot(r, np.log(i1_grid))
plt.plot(r, np.log(k1_grid))
plt.show()

plt.plot(r, mod.source)
plt.show()

plt.plot(r, I1 + K1)
plt.show()

plt.plot(r, integrate.cumtrapz((k1_grid * mod.source)[::-1], dx=sim.grid.dr, initial=0)[::-1])
plt.plot(r, integrate.cumtrapz((i1_grid * mod.source), dx=sim.grid.dr, initial=0))
plt.show()

plt.plot(r, np.log(-((i1_grid * mod.source))))
plt.show()

## seems like trouble...
# try building from separate delta function sources...
sol = np.zeros(r.shape)
for i,rval in enumerate(r):
    source = np.zeros(r.shape)
    source[i] = mod.source[i]
    I1 = i1_grid * integrate.cumtrapz((k1_grid * source)[::-1], dx=sim.grid.dr, initial=0)[::-1]
    K1 = k1_grid * integrate.cumtrapz((i1_grid * source), dx=sim.grid.dr, initial=0)
    sol += I1 + K1

plt.plot(r, mod.source)
plt.show()

plt.plot(r, np.log(I1))
plt.plot(r, np.log(K1+I1))
plt.show()

plt.plot(r, sol)
plt.show()
sol[0] = 2 * sol[1] - 1 * sol[2]
plt.plot(r, sol - sol[0])
plt.show()

