from turbopy import Simulation, Module


class FieldModel(Module):
    """
    This is the abstract base class for field models
    """
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.E = owner.grid.field_factory(3)
        self.B = owner.grid.field_factory(3)
        self.currents = []

    def inspect_resource(self, resource):
        """This modules needs to keep track of current sources"""
        if "ResponseModel:J" in resource:
            print("adding current resource")
            self.currents.append(resource["ResponseModel:J"])

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
        self.J = owner.grid.field_factory(1)  # only Jz for now
        self.E = None

    def exchange_resources(self):
        self.publish_resource({"ResponseModel:J": self.J})

    def inspect_resource(self, resource):
        if "FieldModel:E" in resource:
            print("adding E-field resource")
            self.E = resource["FieldModel:E"]

    def update(self):
        print(self.E[0,0])


Module.add_module_to_library("FieldModel", FieldModel)
Module.add_module_to_library("PlasmaResponseModel", PlasmaResponseModel)


sim_config = {"Modules": [
        {"name": "FieldModel",
         },
        {"name": "PlasmaResponseModel",
         },         
    ],
    "Grid": {"N": 8},
    "Clock": {"start_time": 0,
              "end_time": 3}
    }
    
sim = Simulation(sim_config)
# sim.prepare_simulation()
sim.run()
