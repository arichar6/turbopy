from turbopy import Simulation, Module, Diagnostic, CSVDiagnosticOutput
import numpy as np

class ElectricField(Module):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.E0 = input_data["amplitude"]
        self.E = owner.grid.generate_field()
        self.ddx = owner.find_tool_by_name("FiniteDifference").setup_ddx()
        self.c = 2.998e8
    
    def initialize(self):
        self.E += self.E0 *(1- np.cos(2 * np.pi * self.owner.grid.generate_linear()))
    
    def update(self):
        # need to apply the BC somehow
        self.E[:] = self.E - self.c * self.owner.clock.dt * self.ddx(self.E)
        self.E[0] = self.E[1] - (self.E[2] - self.E[1])
        
    def exchange_resources(self):
        self.publish_resource({"EMField:E": self.E})

class EMWave(Module):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.c = 2.998e8
        self.E0 = input_data["amplitude"]
        self.B0 = self.E0 / self.c
        self.E = owner.grid.generate_field()
        self.B = owner.grid.generate_field()
        self.omega = input_data["omega"]
        self.k = self.omega / self.c
    
    def initialize(self):
        phase = - self.omega * 0 + self.k * self.owner.grid.r
        self.E[:] = self.E0 * np.cos(2 * np.pi * phase)
        self.B[:] = self.B0 * np.sin(2 * np.pi * phase)

    def update(self):
        # need to apply the BC somehow
        phase = - self.omega * self.owner.clock.time + self.k * self.owner.grid.r
        self.E[:] = self.E0 * np.cos(2 * np.pi * phase)
        self.B[:] = self.B0 * np.sin(2 * np.pi * phase)
        
    def exchange_resources(self):
        self.publish_resource({"EMField:E": self.E})
        self.publish_resource({"EMField:B": self.B})
        
class ChargedParticle(Module):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.E = None
        self.B = None
        self.x = input_data["position"]
        self.interp_field = owner.grid.create_interpolator(self.x)
        self.position = np.zeros(3)
        self.momentum = np.zeros(3)
        self.eoverm = 1.7588e11
        self.charge = 1.6022e-19
        self.mass = 9.1094e-31
        
        self.push = owner.find_tool_by_name("BorisPush").push
        
    def update(self):
        E = np.array([0, self.interp_field(self.E), 0])
        B = np.array([0, 0, self.interp_field(self.B)])

        self.push(self.position, self.momentum, self.charge, self.mass, E, B)

    def exchange_resources(self):
        self.publish_resource({"ChargedParticle:position": self.position})
        self.publish_resource({"ChargedParticle:momentum": self.momentum})
    
    def inspect_resource(self, resource):
        if "EMField:E" in resource:
            self.E = resource["EMField:E"]    
        if "EMField:B" in resource:
            self.B = resource["EMField:B"]

class ParticleDiagnostic(Diagnostic):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.data = None
        self.component = input_data["component"]
        
    def inspect_resource(self, resource):
        if "ChargedParticle:" + self.component in resource:
            self.data = resource["ChargedParticle:" + self.component]
    
    def diagnose(self):
        self.output_function(self.data[:])

    def initialize(self):
        # setup output method
        functions = {"stdout": self.print_diagnose,
                     "csv": self.csv_diagnose,
                     }
        self.output_function = functions[self.input_data["output"]]
        if self.input_data["output"] == "csv":
            diagnostic_size = (self.owner.clock.num_steps + 1, 3)
            self.csv = CSVDiagnosticOutput(self.input_data["filename"], diagnostic_size)

    def finalize(self):
        self.diagnose()
        if self.input_data["output"] == "csv":
            self.csv.finalize()

    def print_diagnose(self, data):
        print(data)

    def csv_diagnose(self, data):
        self.csv.append(data)
            

Module.register("EMWave", EMWave)
Module.register("ElectricField", ElectricField)
Module.register("ChargedParticle", ChargedParticle)
Diagnostic.register("ParticleDiagnostic", ParticleDiagnostic)

sim_config = {"Modules": [
        {"name": "EMWave",
         "amplitude": 1,
         "omega": 2e8},
        {"name": "ChargedParticle",
         "position": 0.5},         
    ],
    "Tools": [{"type": "FiniteDifference",
               "method": "upwind_left"},
              {"type": "BorisPush"}
              ],
    "Diagnostics": [
        {"type": "field",
         "component": 0,
         "field": "EMField:E",
         "output": "csv",
         "filename": "efield.csv"},
        {"type": "field",
         "component": 0,
         "field": "EMField:B",
         "output": "csv",
         "filename": "bfield.csv"},
        {"type": "point",
         "field": "EMField:E",
         "location": 0.5,
         "output": "csv",
         "filename": "e_0.5.csv"},
        {"type": "ParticleDiagnostic",
         "output": "csv",
         "component": "momentum",
         "filename": "particle.csv"}
        ],
    "Grid": {"N": 100, "r_min": 0, "r_max": 1},
    "Clock": {"start_time": 0, "end_time": 1e-7, "num_steps": 1000}
    }
    
sim = Simulation(sim_config)
sim.run()
