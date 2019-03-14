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


class CoaxFieldModel(Module):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
    
        r_inner = 0.5e-3  # 1 mm dia rod
        r_outer = 6.0e-3   # 12 mm dia cathode

        mu0_bar = 2.0e-7

        self.E0 = input_data["V0"] / np.log(r_outer/r_inner)
        self.B0 = input_data["I0"] * mu0_bar
        
    def E(self, position):
        r2 = np.dot(position, position)
        rhat = np.array([position[0], position[1], 0])
        return (self.E0 / r2) * rhat

    def B(self, position):
        r2 = np.dot(position, position)
        theta_hat = np.array([-position[1], position[0], 0])
        return (self.B0 / r2) * theta_hat

    def exchange_resources(self):
        self.publish_resource({"FieldModel:E(r)": self.E})
        self.publish_resource({"FieldModel:B(r)": self.B})

    def update(self):
        pass


class Particle:
    def __init__(self, x0 = np.array([0,0,0]), p0 = np.array([0,0,0])):
        self.charge = -1.6022e-19
        self.mass = 9.1094e-31
        self.position = x0.copy()
        self.momentum = p0.copy()
        self.mobile = True
            
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
        x0 = np.array([0.5e-3, 0, 0])
        for i in range(input_data["num_particles"]):
            if "p0" in input_data:
                p0 = input_data["p0"][i]
            else:
                p0 = np.zeros(3)
            self.particles.append(Particle(p0=p0, x0=x0))
        
    def exchange_resources(self):
        self.publish_resource({"ParticlePusher:particles": self.particles})

    def inspect_resource(self, resource):
        if "FieldModel:E(r)" in resource:
            print("adding E-field resource")
            self.E = resource["FieldModel:E(r)"]
        if "FieldModel:B(r)" in resource:
            print("adding B-field resource")
            self.B = resource["FieldModel:B(r)"]

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
            if p.mobile:
                E = self.E(p.position)
                B = self.B(p.position)
            
                vminus = p.momentum + dt * E * p.charge / 2
                m1 = np.sqrt(p.mass**2 + np.dot(p.momentum, p.momentum)/c2)
                t = dt * B * p.charge / m1 / 2
                s = 2 * t / (1 + np.dot(t, t))
                vprime = vminus + np.cross(vminus, t)
                vplus = vminus + np.cross(vprime, s)
                p.momentum = vplus + dt * E * p.charge / 2
                m2 = np.sqrt(p.mass**2 + np.dot(p.momentum, p.momentum)/c2)
                p.position = p.position + dt * p.momentum / m2
                
                if (p.position[0] * p.position[0] + p.position[1] * p.position[1]) < (0.5e-3)**2:
                    p.mobile = False
        
Module.add_module_to_library("ConstantFieldModel", ConstantFieldModel)
Module.add_module_to_library("ParticlePusher", ParticlePusher)
Module.add_module_to_library("CoaxFieldModel", CoaxFieldModel)

class ParticleDiagnostic(Diagnostic):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)

        self.output = input_data["output"] # "stdout"
        self.particle_list = []
        self.file = None
        self.npart = 0
        
    def diagnose(self):
        self.output_function(self.particle_list)

    def inspect_resource(self, resource):
        if "ParticlePusher:particles" in resource:
            self.particle_list = resource["ParticlePusher:particles"]
    
    def initialize(self):
        # setup output method
        functions = {"stdout": print,
                     "csv": self.write_to_csv,
                     }
        self.output_function = functions[self.input_data["output"]]
        self.npart = len(self.particle_list)
        if self.input_data["output"] == "csv":
            self.outputbuffer = np.zeros((
                        self.owner.clock.num_steps+1,
                        6 * self.npart
                        ))
    
    def write_to_csv(self, data):
        i = self.owner.clock.this_step
        for j in range(self.npart):
            self.outputbuffer[i,0 + 6*j:3 + 6*j] = data[j].position
            self.outputbuffer[i,3 + 6*j:6 + 6*j] = data[j].momentum
    
    def finalize(self):
        self.diagnose()
        if self.input_data["output"] == "csv":
            self.file = open(self.input_data["filename"], 'wb')
            np.savetxt(self.file, self.outputbuffer, delimiter=",")
            self.file.close()


Diagnostic.add_diagnostic_to_library("particle", ParticleDiagnostic)

dt = 1e-14
t1 = 3e-12
num_steps = int(t1/dt)

sim_config = {"Modules": [
        {"name": "CoaxFieldModel",
            "V0": 2.4e6,
            "I0": 1e5,
        },
        {"name": "ParticlePusher",
            "num_particles": 5,
            "p0": np.array([[1e5, 0, 0],
                            [2e5, 0, 0],
                            [8.6e5, 0, 0],
                            [1.5e6, 0, 0],
                            [2.345e6, 0, 0],]) * 1.6022e-19 / 2.9979e8
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
              "end_time":  t1, 
              "num_steps": num_steps,
              }
    }
    
sim = Simulation(sim_config)
sim.run()