from turbopy import Simulation, Module, Diagnostic

class MyModule(Module):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        print("MyModule Loaded: ", input_data["param"])
        self.data = [input_data["param"], 0]
    
    def update(self):
        self.data[1] = self.data[0] * self.owner.clock.time

    def exchange_resources(self):
        self.publish_resource({"MyModule:data": self.data})


class MyDiagnostic(Diagnostic):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.data = None
        
    def inspect_resource(self, resource):
        if "MyModule:data" in resource:
            self.data = resource["MyModule:data"]
    
    def diagnose(self):
        print("Time", self.owner.clock.time)
        print("Data", self.data)
    
    def initialize(self):
        print("Initializing diagnostic...")
        self.diagnose()
    
    def finalize(self):
        print("Finalizing diagnostic...")
        self.diagnose()

Module.register("MyModule", MyModule)
Diagnostic.register("MyDiagnostic", MyDiagnostic)

sim_config = {"Modules": [
        {"name": "MyModule",
         "param": 3.14},
        {"name": "MyModule",
         "param": 22},         
    ],
    "Diagnostics": [{"type": "MyDiagnostic"}],
    "Grid": {"N": 8, "r_min": 0, "r_max": 1},
    "Clock": {"start_time": 0, "end_time": 3, "num_steps": 3}
    }
    
sim = Simulation(sim_config)
sim.run()
