from turbopy import Simulation, Module
import numpy as np

class FieldModel(Module):
    """
    This is the abstract base class for field models
    """
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.E = owner.grid.generate_field(3)
        self.B = owner.grid.generate_field(3)
        self.currents = []

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
        self.E[0,0] = self.owner.clock.time


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
                        self.peak_current,
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
        {"name": "FieldModel",
         },
        {"name": "PlasmaResponseModel",
         },
        {"name": "RigidBeamCurrentSource",
         "peak_current": 1.0e5,
         "beam_radius": 0.01,
         "rise_time": 30.0e-9,
         "profile": "gaussian",
         },
    ],
    "Diagnostics": [
        {"type": "point",
         "location": 0.005,
         "field": "FieldModel:E",
         "output": "stdout",
         }
    ],
    "Grid": {"N": 8,
             "r_min": 0, "r_max": 0.1,
            },
    "Clock": {"start_time": 0,
              "end_time": 30e-9, 
              "num_steps": 5,
              }
    }
    
sim = Simulation(sim_config)
# sim.prepare_simulation()
sim.run()
