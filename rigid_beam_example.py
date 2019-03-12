from turbopy import Simulation, Module
import numpy as np

class FieldModel(Module):
    """
    This is the abstract base class for field models
    """
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.mu0 = 4 * np.pi * 1e-7
        self.E = owner.grid.generate_field(1)
        self.B = owner.grid.generate_field(1)
        self.currents = []
        self.solver_name = input_data["solver"]
        self.solver = None
        
        self.sourceterm = owner.grid.generate_field(1)
        self.old_source = owner.grid.generate_field(1)

    def initialize(self):
        self.solver = self.owner.find_tool_by_name(self.solver_name)

    def inspect_resource(self, resource):
        """This modules needs to keep track of current sources"""
        if "ResponseModel:J" in resource:
            print("adding current from plasma response")
            self.currents.append(resource["ResponseModel:J"])
        if "CurrentSource:J" in resource:
            print("adding current source")
            self.currents.append(resource["CurrentSource:J"])
            
    def exchange_resources(self):
        """Tell other modules about the electric field, in case the need it"""
        self.publish_resource({"FieldModel:E": self.E})
        self.publish_resource({"FieldModel:B": self.B})

    def update(self):
        self.old_source[:] = self.sourceterm[:]
        self.sourceterm[:] = np.sum(self.currents, axis=0)
        dt = self.owner.clock.dt
        dJ = (self.sourceterm - self.old_source)/dt
        self.E[:,0] = self.solver.solve( self.mu0 * dJ[:,0] )


class PlasmaResponseModel(Module):
    """
    This is the abstract base class for plasma response models
    """
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.J = owner.grid.generate_field(1)  # only Jz for now
        self.E = None

    def exchange_resources(self):
        self.publish_resource({"ResponseModel:J": self.J})

    def inspect_resource(self, resource):
        if "FieldModel:E" in resource:
            print("adding E-field resource")
            self.E = resource["FieldModel:E"]

    def update(self):
        pass


class RigidBeamCurrentSource(Module):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.input_data = input_data
        self.peak_current = input_data["peak_current"]
        self.beam_radius = input_data["beam_radius"]
        self.rise_time = input_data["rise_time"]

        self.J = owner.grid.generate_field(1)           # only Jz for now
        self.profile = owner.grid.generate_field(1)
        self.set_profile(input_data["profile"])
                
    def exchange_resources(self):
        self.publish_resource({"CurrentSource:J": self.J})
    
    def set_profile(self, profile_type):
        profiles = {"gaussian": lambda r: 
                        self.peak_current * np.exp(-(r/self.beam_radius)**2),
                    "uniform": lambda r:
                        self.peak_current * (r < self.beam_radius),
                    "bennett": lambda r:
                        self.peak_current * 1.0/(1.0+(r/self.beam_radius)**2)**2,
                    }
        try:
            self.profile[:, 0] = profiles[profile_type](self.owner.grid.r)
        except KeyError:
            raise KeyError("Unknown profile type: {0}".format(profile_type))
            
    def update(self):
        self.set_current_for_time(self.owner.clock.time)
    
    def set_current_for_time(self, time):
        self.J[:] = np.sin(np.pi*time/self.rise_time/2)**2 * self.profile        


Module.add_module_to_library("FieldModel", FieldModel)
Module.add_module_to_library("PlasmaResponseModel", PlasmaResponseModel)
Module.add_module_to_library("RigidBeamCurrentSource", RigidBeamCurrentSource)


sim_config = {"Modules": [
        {"name": "PlasmaResponseModel",
         },
        {"name": "RigidBeamCurrentSource",
             "peak_current": 1.0e5,
             "beam_radius": 0.05,
             "rise_time": 30.0e-9,
             "profile": "uniform",
         },
        {"name": "FieldModel",
             "solver": "PoissonSolver1DRadial",
         },
    ],
    "Diagnostics": [
        {"type": "field",
             "field": "FieldModel:E",
             "output": "csv",
             "filename": "efield.csv",
             "component": 0,
         },
        {"type": "field",
             "field": "CurrentSource:J",
             "output": "csv",
             "filename": "currentsource.csv",
             "component": 0,
         },
         {"type": "grid"},
    ],
    "Tools": [
        {"type": "PoissonSolver1DRadial",
         }],
    "Grid": {"N": 32,
                 "r_min": 0, "r_max": 0.1,
            },
    "Clock": {"start_time": 0,
                  "end_time": 30e-9, 
                  "num_steps": 25,
              }
    }
    
sim = Simulation(sim_config)
sim.run()


