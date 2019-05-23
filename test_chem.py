# import chemistry as ch
# 
# c = ch.Chemistry(["chemistry/N2_Rates_TT.txt"])
# 
# 
# print([s.name for s in c.species])
# print("There are", len([s.name for s in c.species]), "unique species")


# test the FD operators
from turbopy import Simulation, Module
import numpy as np
from scipy import sparse

sim_config = {"Modules": [
    ],
    "Diagnostics": [
    ],
    "Tools": [
        {"type": "FiniteDifference",
         }],
    "Grid": {"N": 20 ,
             "r_min": 0, "r_max": 1,
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
J[:] = np.exp(-(r/0.5)**2)

fd = sim.find_tool_by_name("FiniteDifference")
D = fd.del2_radial()
ddr = fd.ddr()
BC_right = fd.BC_right_extrap()
BC_left = fd.BC_left_extrap()
BCl2 = fd.BC_left_avg()

# print(BCl2.toarray())

import matplotlib.pyplot as plt

inner = BC_right @ (r * (BC_left @ (ddr @ (J))))
rinv = (1/r)
rinv[0] = 0

plt.subplot(3,1,1)
plt.plot(r, inner, '.-')
plt.subplot(3,1,2)
plt.plot(r, BC_right @ (rinv * (ddr @ inner)), '.-')
plt.subplot(3,1,3)
plt.plot(r, (BCl2 @ (BC_right @ ( rinv *  (ddr @ inner)))), '.-')
plt.show()

