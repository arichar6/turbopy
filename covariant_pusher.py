from turbopy import Simulation, Module, Diagnostic
import numpy as np
import mcpl
from numpy import linalg
np.set_printoptions(precision=3)

class ConstantFieldModel(Module):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
    
        self.E0 = input_data["E0"] * np.array([1, 0, 0])
        self.B0 = input_data["B0"] * np.array([0, 1, 0])

    def E(self, x):
        return self.E0
    
    def B(self, x):
        return self.B0

    def exchange_resources(self):
        self.publish_resource({"FieldModel:E(r)": self.E})
        self.publish_resource({"FieldModel:B(r)": self.B})

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
        r2 = (position[0]**2 + position[1]**2)
        rhat = np.array([position[0], position[1], 0])
        return (self.E0 / r2) * rhat

    def B(self, position):
        r2 = (position[0]**2 + position[1]**2)
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


class MCPLParticleLoader(Module):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.filename = input_data['filename']
        self.nmax = input_data['nmax']
        self.particles = None
        
    def inspect_resource(self, resource):
        if "ParticlePusher:particles" in resource:
            self.particles = resource["ParticlePusher:particles"]

    def initialize(self):
        myfile = mcpl.MCPLFile(self.filename)
        m2c2 = ((9.1094e-31) * (2.9979e8))**2
        nmax = self.nmax
        if nmax < 0:
            nmax = myfile.nparticles
        for i, p in enumerate(myfile.particles):
            dir = p.direction
            gamma = (p.ekin / 0.511) + 1
            pmag = (m2c2 * (gamma**2 - 1))**0.5
            part = Particle(p.position / 100, p.direction * pmag)
            # print("position:", p.position, "momentum:", p.direction * pmag)
            self.particles.append(part)
            if i == nmax-1:
                return
    
    def update(self):
        pass


class ParticlePusher(Module):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        
        self.particles = []
        
        self.E = None
        self.B = None
        
        self.create_particles(input_data)
        
        self.update = self.gordon_push

    def create_particles(self, input_data):
        for i in range(input_data["num_particles"]):
            if "p0" in input_data:
                p0 = np.array(input_data["p0"][i])
            else:
                p0 = np.zeros(3)
            if "x0" in input_data:
                x0 = np.array(input_data["x0"][i])
            else:
                x0 = np.array([0.5e-3, 0, 0])
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

    def initialize(self):
        c2 = (2.9979e8)**2
        for p in self.particles:
            if p.mobile:
                m2 = np.sqrt(p.mass**2 + np.dot(p.momentum, p.momentum)/c2)
                p.position = p.position + dt * p.momentum / m2 / 2

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

    def fields_matrix(self, E, B):
        c = 2.9979e8
        return np.array([[0, -E[0]/c, -E[1]/c, -E[2]/c],
                         [E[0]/c, 0, -B[2], B[1]],
                         [E[1]/c, B[2], 0, -B[0]],
                         [E[2]/c, -B[1], B[0], 0]])

    def calc_new_momentum(self, position, momentum, mass, charge):
        E = self.E(position)
        B = self.B(position)
        c1 = 2.9979e8
        c2 = c1**2
        gamma = np.sqrt(1 + np.dot(momentum, momentum)/mass**2 / c2)
        F_mat = dt * charge * self.fields_matrix(E,B) / mass / gamma
        d_mat, rot_mat = linalg.eig(F_mat)
        rot_inv = linalg.inv(rot_mat)
        new_mat = rot_mat @ np.diag(np.exp(d_mat)) @ rot_inv  # use numpy @-operator for matrix multiply
                        
        return new_mat        

    def gordon_push(self):
        dt = self.owner.clock.dt
        c1 = 2.9979e8
        c2 = c1**2
        for p in self.particles:
            if p.mobile:
                new_mat = self.calc_new_momentum(p.position, p.momentum, p.mass, p.charge)
                gmc =  np.sqrt(p.mass**2 * c2 + np.dot(p.momentum, p.momentum))
                four_momentum = np.array([gmc, p.momentum[0], p.momentum[1], p.momentum[2]])
                new_mom = (new_mat @ four_momentum)[1:]

                m2 = np.sqrt(p.mass**2 + np.dot(new_mom, new_mom)/c2)
                x1 = p.position + dt * new_mom / m2
                new_mat2 = self.calc_new_momentum(x1, new_mom, p.mass, p.charge)

                # p.momentum[:] = (((new_mat + new_mat2)/2) @ four_momentum )[1:]
                # p.momentum[:] = (new_mat @ four_momentum )[1:]
                p.momentum[:] = (new_mat2 @ four_momentum )[1:]
                
                m2 = np.sqrt(p.mass**2 + np.dot(p.momentum, p.momentum)/c2)
                p.position = p.position + dt * p.momentum / m2
                
                if (p.position[0] * p.position[0] + p.position[1] * p.position[1]) < (0.5e-3)**2:
                    p.mobile = False                

    def higuera_cary_push(self):
        raise(NotImplementedError("The Higuera-Cary pusher has not been implemented"))
        dt = self.owner.clock.dt
        c2 = (2.9979e8)**2
        for p in self.particles:
            if p.mobile:
                m2c2 = p.mass**2 * c2
                E = self.E(p.position)
                B = self.B(p.position)
            
                vi = None
                gamma2 = 1 + np.dot(p.momentum, p.momentum) / m2c2

                m2 = np.sqrt(p.mass**2 + np.dot(p.momentum, p.momentum)/c2)
                p.position = p.position + dt * p.momentum / m2
                
                if (p.position[0] * p.position[0] + p.position[1] * p.position[1]) < (0.5e-3)**2:
                    p.mobile = False


Module.add_module_to_library("ConstantFieldModel", ConstantFieldModel)
Module.add_module_to_library("ParticlePusher", ParticlePusher)
Module.add_module_to_library("CoaxFieldModel", CoaxFieldModel)
Module.add_module_to_library("MCPLParticleLoader", MCPLParticleLoader)


class ParticleDiagnostic(Diagnostic):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)

        self.output = input_data["output"] # "stdout"
        self.particle_list = []
        self.file = None
        self.npart = 1
        
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
        # self.npart = len(self.particle_list)
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


class ParticleHistogram(Diagnostic):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.particles = None
    
    def inspect_resource(self, resource):
        if "ParticlePusher:particles" in resource:
            self.particles = resource["ParticlePusher:particles"]
    
    def diagnose(self):
        pass

    def initialize(self):
        print("writing initial z-position histogram")
        zvals = np.zeros(len(self.particles))
        for i,p in enumerate(self.particles):
            zvals[i] = p.position[2]
        h, bin = np.histogram(zvals, bins=30, range=(-5/1000, 5/1000))
        h.tofile("h_pre.csv", ",")
        bin.tofile("bin_pre.csv", ",")
            
    def finalize(self):
        print("writing final z-position histogram")
        zvals = np.zeros(len(self.particles))
        for i,p in enumerate(self.particles):
            zvals[i] = p.position[2]
        h, bin = np.histogram(zvals, bins=30, range=(-5/1000, 5/1000))
        h.tofile("h_post.csv", ",")
        bin.tofile("bin_post.csv", ",")

Diagnostic.add_diagnostic_to_library("particle", ParticleDiagnostic)
Diagnostic.add_diagnostic_to_library("particle_histogram", ParticleHistogram)

dt = 3e-14
t1 = 3e-12
#t1 = dt
num_steps = int(t1/dt)

sim_config = {"Modules": [
        {"name": "CoaxFieldModel",
            "V0": 2.4e6, # 2.4e6
            "I0": -1e5,   # 1e5
        },
#         {"# name": "ConstantFieldModel",
#             "E0": 0, 
#             "B0": 50,
#         },
#         {"name": "ParticlePusher",
#             "num_particles": 1,
#             "p0":[ [-2.345e6 * 1.6022e-19 / 2.9979e8, 0, 0] ],
#             "x0":[ [-0.05/100,  0.000,  0.075/100] ],
#         },
        {"name": "ParticlePusher",
            "num_particles": 0,
        },
        {"name": "MCPLParticleLoader",
            "filename": "output.mcpl",
            "nmax": -1
        },
#         {"name": "ParticlePusher",
#             "num_particles": 5,
#             "p0": np.array([[1e5, 0, 0],
#                             [2e5, 0, 0],
#                             [8.6e5, 0, 0],
#                             [1.5e6, 0, 0],
#                             [2.345e6, 0, 0],]) * 1.6022e-19 / 2.9979e8
#         },
    ],
    "Diagnostics": [
        {"type": "particle",
         "output": "csv",
         "filename": "particle.csv",
        },
        {"type": "particle_histogram"
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