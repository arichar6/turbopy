from turbopy import Simulation, Module
import numpy as np
import chemistry as Chem

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
        
        self.update = self.forward_integrate
        
        

    def exchange_resources(self):
        self.publish_resource({"ResponseModel:J": self.J})

    def inspect_resource(self, resource):
        if "FieldModel:E" in resource:
            print("adding E-field resource")
            self.E = resource["FieldModel:E"]

    def forward_integrate():
        pass
#        
class ThermalFluidPlasma(Module):
    """
    This class handles the themal plasma (fluid-like) response that is used to 
    compute plasma return currents.  The only required function is update_J which
    provides the plasma return current.
    q                    - is the species charge
    m                    - is the species mass
    list_of_rate_files   - is the path to the chemistry file containing rate coefficients
    chem_dir             - is the path directory where gas_chemistry.py resides
        """
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.J = owner.grid.generate_field(1)  # only Jz for now
        self.E = None
#        
# We need to setup the chemistry and establish the initial conditions in the plasma
# fluid.  
#
#  Establish an instance to the Chemistry class and define some constants
#
        self.echarge = 1.602E-19                     # magnitude of the electron charge in Coulombs
        self.AMU     = 1.67377E-27                   # mass of 1 Atomic Mass Unit in kg
#
#  The call to plasma_chemistry(list_of_rate_files) reads the establishes the plasma chemistry data structure 
#  and establishes a unique set of species for all of the reactions listed in the rate-equation file(s).  It also 
#  sets up a dictionary which tells the class which species are products and which are reactants for
#  each reaction in the file.
#
        path_to_rates = input_data["RateFileList"] 
        initial_conditions= input_data["initial_conditions"]
        self.chem = Chem.Chemistry(path_to_rates) 
        self.species = self.chem.species
#  Set the initial conditions on the plasma fluid
        self.set_initial_plasma_quantities(initial_conditions, owner)
        self.rhs = self.chem.RHS(self.species)
        self.update = self.forward_Euler

    def set_initial_plasma_quantities(self, initial_conditions, owner):
        """
        This function determines the shape and size of the grid and sets the 
        initial conditions for each specties accordingly. 
        
        initial_conditions   - a dictionary of species that have non-zero partial pressures at t=0
        
        """
        NLoschmidt = 2.686E25                        # This is the number of molecules in an ideal gas at stp
        pressure = initial_conditions['pressure']
        velocity = initial_conditions['velocity']
        erg = initial_conditions['erg']
        initial_species = erg.keys()
        
        for species_name in self.species.keys():
            if species_name in initial_species:
                density = NLoschmidt*pressure[species_name]
                V = velocity[species_name]
                energy = erg[species_name]
                self.species[species_name].density  = density  * (1 + owner.grid.generate_field(1))
                self.species[species_name].velocity = V * (1 + owner.grid.generate_field(1))
                self.species[species_name].energy   = energy * (1 + owner.grid.generate_field(1))
                print ( '    initial conditions have been set for species', species_name )
            else:
                self.species[species_name].density  = owner.grid.generate_field(1)
                self.species[species_name].velocity = owner.grid.generate_field(1)
                self.species[species_name].energy   = owner.grid.generate_field(1)
        return 

    def predictor_corrector(self):
        """
        This function performs a predictor-corrector for updating the densities,
        the electron velocity and energy
        """
        dt = self.owner.clock.dt
#  This is the predictor step
        sp_predictor = self.integrator(self.species, 0.5*dt)
#  This is the corrector step
        self.species = self.integrator(sp_predictor, dt)

        return
        
    def integrator(self, species, dt):
        sp_copy = species.copy()
        rhs = self.chem.RHS(sp_copy)
        for s in species.keys():
            sp_copy[s].density = species[s].density + rhs['density'][s] * dt
        sp_copy['e'].velocity = species['e'].velocity - rhs['nu_m']['e']*species['e'].velocity* dt
        sp_copy['e'].energy = species['e'].energy   + rhs['energy']['e'] * dt
        return sp_copy
        
    def forward_Euler(self):
        dt = self.owner.clock.dt
        
        Jp = 0.0
        rhs = self.chem.RHS(self.species)
        self.rhs = rhs
        for sp in self.species.keys():
            self.species[sp].density[:]  = self.species[sp].density + rhs['density'][sp]*dt
            M = self.AMU*self.species[sp].A
            q = self.echarge*self.species[sp].Z
#            F = self.species[sp].q*(E+self.cross_product(self.species[sp],B))
            F = q*self.E
            nu_m = rhs['nu_m'][sp]
            VDrift = F/(M*nu_m)
#            self.species[sp].velocity[:]=VDrift
            self.species[sp].velocity[:] = self.species[sp].velocity + (F/M - nu_m*self.species[sp].velocity)*dt
#            self.species[sp].energy = self.species[sp].energy + (np.dot(self.species[sp].velocity,self.E)-rhs['energy'][sp])*dt
            self.species[sp].energy[:] = self.species[sp].energy + (self.species[sp].velocity*self.E+rhs['energy'][sp])*dt
            Jp = Jp + q*self.species[sp].density*self.species[sp].velocity
        self.J[:] = Jp
        return

    def exchange_resources(self):
        self.publish_resource({"ResponseModel:J": self.J})
        self.publish_resource({"ElectronEnergy": self.species['e'].energy})
        self.publish_resource({"ElectronVelocity": self.species['e'].velocity})
        self.publish_resource({"ElectronDensity": self.species['e'].density})
        self.publish_resource({"GasDensity": self.species['N2(X1)'].density})
        self.publish_resource({"MomentumTransfer": self.rhs['nu_m']['e']})
        
        

    def inspect_resource(self, resource):
        if "FieldModel:E" in resource:
            print("adding E-field resource")
            self.E = resource["FieldModel:E"]


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
Module.add_module_to_library("ThermalFluidPlasma", ThermalFluidPlasma)
Module.add_module_to_library("RigidBeamCurrentSource", RigidBeamCurrentSource)
##
#  Chemistry files
RateFileList=['N:\\Codes\\turbopy\\chemistry\\N2_Rates_TT.txt']
# Initial Conditions:
#       All species that do not have a non-zero partial pressure at t=0 will be set to zero
#
initial_conditions={"pressure":{"N2(X1)":1.0/760,"e":1e-5*1.0/760},
                    "velocity":{"N2(X1)":0.0,    "e":0.0},
                    "erg":{     "N2(X1)":1.0/40.,"e":0.1}  }
                    
end_time = 30.E-9
dt = 1.0E-9
number_of_steps = int(end_time/dt)
N_grid = 8

sim_config = {"Modules": [
        {"name": "FieldModel",
             "solver": "PoissonSolver1DRadial",
         },
        {"name": "ThermalFluidPlasma",
        "RateFileList":RateFileList,
        "initial_conditions":initial_conditions
         },
        {"name": "RigidBeamCurrentSource",
             "peak_current": 1.0e5,
             "beam_radius": 0.05,
             "rise_time": 30.0e-9,
             "profile": "uniform",
         },
    ],
    "Diagnostics": [
        {"type": "field",
             "field": "FieldModel:E",
             "output": "csv",
             "filename": "Efield.csv",
             "component": 0,
         },
        {"type": "field",
             "field": "CurrentSource:J",
             "output": "csv",
             "filename": "BeamCurrent.csv",
             "component": 0,
         },
        {"type": "field",
             "field": "ResponseModel:J",
             "output": "csv",
             "filename": "PlasmaCurrent.csv",
             "component": 0,
         },
        {"type": "field",
             "field": "ElectronEnergy",
             "output": "csv",
             "filename": "ElectronEnergy.csv",
             "component": 0,
         },
        {"type": "field",
             "field": "ElectronVelocity",
             "output": "csv",
             "filename": "ElectronVelocity.csv",
             "component": 0,
         },
        {"type": "field",
             "field": "ElectronDensity",
             "output": "csv",
             "filename": "ElectronDensity.csv",
             "component": 0,
         },
        {"type": "field",
             "field": "GasDensity",
             "output": "csv",
             "filename": "GasDensity.csv",
             "component": 0,
         },
        {"type": "field",
             "field": "MomentumTransfer",
             "output": "csv",
             "filename": "nuM.csv",
             "component": 0,
         },
            {"type": "grid"},
    ],
    "Tools": [
        {"type": "PoissonSolver1DRadial",
         }],
    "Grid": {"N":N_grid ,
             "r_min": 0, "r_max": 0.1,
            },
    "Clock": {"start_time": 0,
              "end_time":  end_time, 
              "num_steps": number_of_steps,

              }
    }
    
sim = Simulation(sim_config)
#
sim.run()