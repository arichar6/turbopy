from turbopy import Simulation, Module

class MyModule(Module):
    def __init__(self, owner, input_data):
        super().__init__(owner, input_data)
        print("MyModule Loaded")

Module.add_module_to_library("MyModule", MyModule)

sim_config = {"Modules": [
        {"name": "MyModule",
         "more_config_options": 3.14,
         },
    ],
    "Grid": {},
    }
    
sim = Simulation(sim_config)
sim.prepare_simulation()

print("Loaded modules are: ", sim.modules)