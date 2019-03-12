from turbopy import Simulation, Module

class MyModule(Module):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        print("MyModule Loaded: ", input_data["param"])
        self.x = input_data["param"]
    
    def update(self):
        print(self.owner.clock.time * self.x)
    
    def exchange_resources(self):
        self.publish_resource({"MyModule:x": self.x})
    
    def inspect_resource(self, resource):
        for k,v in resource.items():
            print("I don't need:", k, v)

Module.add_module_to_library("MyModule", MyModule)

sim_config = {"Modules": [
        {"name": "MyModule",
         "param": 3.14,
         },
        {"name": "MyModule",
         "param": 22,
         },         
    ],
    "Grid": {"N": 8},
    "Clock": {"start_time": 0,
              "end_time": 3}
    }
    
sim = Simulation(sim_config)
sim.run()
