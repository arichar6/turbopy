from turbopy import Simulation, Module
import numpy as np


class ConstantFieldModel(Module):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
    
        self.E = input_data["E0"] + owner.grid.generate_field(1)
        self.B = input_data["B0"] + owner.grid.generate_field(1)

    def exchange_resources(self):
        self.publish_resource({"FieldModel:E": self.E})
        self.publish_resource({"FieldModel:B": self.B})

    def update(self):
        pass


class Particle:
    def __init__(self):
        self.charge = -1.6022e-19
        self.mass = 9.1094e-31
        self.position = np.zeros(3)
        self.momentum = np.zeros(3)
        
class ParticlePusher(Module):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        
        self.particles = []
        
        self.E = None
        self.B = None
        
        self.create_particles(input_data)

    def create_particles(self, input_data):
        for i in range(input_data["num_particles"]):
            self.particles.append(Particle())
        
    def exchange_resources(self):
        self.publish_resource({"ParticlePusher:particles": self.particles})

    def inspect_resource(self, resource):
        if "FieldModel:E" in resource:
            print("adding E-field resource")
            self.E = resource["FieldModel:E"]

    def update(self):
        pass

        
Module.add_module_to_library("ConstantFieldModel", ConstantFieldModel)
Module.add_module_to_library("ParticlePusher", ParticlePusher)

sim_config = {"Modules": [
        {"name": "ConstantFieldModel",
            "E0": 1e4,
            "B0": 1e4,
        },
        {"name": "ParticlePusher",
            "num_particles": = 5,
        },
    ],
    "Diagnostics": [
        {"type": "field",
             "field": "FieldModel:E",
             "output": "csv",
             "filename": "Efield.csv",
             "component": 0,
         },
         {"type": "grid"},
    ],
    "Tools": [
        {"type": "PoissonSolver1DRadial",
         }],
    "Grid": {"N":8 ,
             "r_min": 0, "r_max": 0.1,
            },
    "Clock": {"start_time": 0,
              "end_time":  30e-9, 
              "num_steps": 10,

              }
    }
    
sim = Simulation(sim_config)
sim.run()