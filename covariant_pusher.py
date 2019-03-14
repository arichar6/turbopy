from turbopy import Simulation, Module, Diagnostic
import numpy as np


class ConstantFieldModel(Module):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
    
        self.E = input_data["E0"] * np.array([1, 0, 0])
        self.B = input_data["B0"] * np.array([0, 1, 0])

    def exchange_resources(self):
        self.publish_resource({"FieldModel:E": self.E})
        self.publish_resource({"FieldModel:B": self.B})

    def update(self):
        pass


class Particle:
    def __init__(self, x0 = np.array([0,0,0]), p0 = np.array([0,0,0])):
        self.charge = -1.6022e-19
        self.mass = 9.1094e-31
        self.position = x0
        self.momentum = p0
            
    def __repr__(self):
        return "x: " + str(self.position) + ", v: " + str(self.momentum)
        
class ParticlePusher(Module):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        
        self.particles = []
        
        self.E = None
        self.B = None
        
        self.create_particles(input_data)
        
        self.update = self.boris_push

    def create_particles(self, input_data):
        if "p0" in input_data:
            p0 = input_data["p0"]
        else:
            p0 = np.zeros(3)
        for i in range(input_data["num_particles"]):
            self.particles.append(Particle(p0=p0))
        
    def exchange_resources(self):
        self.publish_resource({"ParticlePusher:particles": self.particles})

    def inspect_resource(self, resource):
        if "FieldModel:E" in resource:
            print("adding E-field resource")
            self.E = resource["FieldModel:E"]
        if "FieldModel:B" in resource:
            print("adding B-field resource")
            self.B = resource["FieldModel:B"]

    def euler_push(self):
        dt = self.owner.clock.dt
        # update the position and velocity of the particles based on Lorenz force
        for p in self.particles:
            force = p.charge * (self.E + np.cross(p.momentum, self.B) / p.mass)
            p.momentum = p.momentum + dt * force
            velocity = p.momentum / p.mass
            p.position = p.position + dt * velocity
    
    def boris_push(self):
        dt = self.owner.clock.dt
        c2 = (2.9979e8)**2
        for p in self.particles:
            vminus = p.momentum + dt * self.E * p.charge / 2
            m1 = np.sqrt(p.mass**2 + np.dot(p.momentum, p.momentum)/c2)
            t = dt * self.B * p.charge / m1 / 2
            s = 2 * t / (1 + np.dot(t, t))
            vprime = vminus + np.cross(vminus, t)
            vplus = vminus + np.cross(vprime, s)
            p.momentum = vplus + dt * self.E * p.charge / 2
            m2 = np.sqrt(p.mass**2 + np.dot(p.momentum, p.momentum)/c2)
            p.position = p.position + dt * p.momentum / m2
                
        
Module.add_module_to_library("ConstantFieldModel", ConstantFieldModel)
Module.add_module_to_library("ParticlePusher", ParticlePusher)


class ParticleDiagnostic(Diagnostic):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)

        self.output = input_data["output"] # "stdout"
        self.particle_list = []
        self.file = None
        
    def diagnose(self):
        self.output_function(self.particle_list[0])

    def inspect_resource(self, resource):
        if "ParticlePusher:particles" in resource:
            self.particle_list = resource["ParticlePusher:particles"]
    
    def initialize(self):
        # setup output method
        functions = {"stdout": print,
                     "csv": self.write_to_csv,
                     }
        self.output_function = functions[self.input_data["output"]]
        if self.input_data["output"] == "csv":
            self.outputbuffer = np.zeros((
                        self.owner.clock.num_steps+1,
                        6
                        ))
    
    def write_to_csv(self, data):
        i = self.owner.clock.this_step
        self.outputbuffer[i,:3] = data.position
        self.outputbuffer[i,3:] = data.momentum
    
    def finalize(self):
        self.diagnose()
        if self.input_data["output"] == "csv":
            self.file = open(self.input_data["filename"], 'wb')
            np.savetxt(self.file, self.outputbuffer, delimiter=",")
            self.file.close()


Diagnostic.add_diagnostic_to_library("particle", ParticleDiagnostic)



sim_config = {"Modules": [
        {"name": "ConstantFieldModel",
            "E0": -1e5,
            "B0": -1,
        },
        {"name": "ParticlePusher",
            "num_particles": 1,
            "p0": 9.1094e-31 * np.array([1e6, 0, 0])
        },
    ],
    "Diagnostics": [
        {"type": "particle",
         "output": "csv",
         "filename": "particle.csv",
        },
    ],
    "Tools": [
        {"type": "PoissonSolver1DRadial",
         }],
    "Grid": {"N":8 ,
             "r_min": 0, "r_max": 0.1,
            },
    "Clock": {"start_time": 0,
              "end_time":  1e-10, 
              "num_steps": 100,

              }
    }
    
sim = Simulation(sim_config)
sim.run()