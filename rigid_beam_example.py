from turbopy import Simulation, Module, Diagnostic
import numpy as np
from chemistry import Species, Chemistry

from pathlib import Path

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
        # E_Td = 13.61
        # E_Td = 211.9 
        # self.E0 = 1E-21*E_Td*3.53420895e+22
        # print (self.E0)
        # self.owner = owner
        # self.E[:] = self.E0 * (1 + self.owner.grid.generate_field(1) )

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
        # return
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
class FluidSpecies:
    def __init__(self, species, mobile_species, owner, density=0, velocity=0, energy=0):
        self.q = species.charge
        self.m = species.mass
        self.name = species.name
        self.density = density*(1+owner.grid.generate_field(1))
        if mobile_species:
            self.nV = self.density*velocity*(1+owner.grid.generate_field(1))
            self.nEnergy   = self.density*energy*(1+owner.grid.generate_field(1))
        self.mobile_species = mobile_species        
        
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
        self.J = self.owner.grid.generate_field(1)  # only Jz for now
        self.E = None
#        
# We need to setup the chemistry and establish the initial conditions in the plasma
# fluid.  
#
#  Establish an instance to the Chemistry class and define some constants
#
#
#  The call to plasma_chemistry(list_of_rate_files) reads the establishes the plasma chemistry data structure 
#  and establishes a unique set of species for all of the reactions listed in the rate-equation file(s).  It also 
#  sets up a dictionary which tells the class which species are products and which are reactants for
#  each reaction in the file.
#
#  Set the initial conditions on the plasma fluid
        self.update = self.forward_Euler
        self.echarge = 1.602E-19

        path_to_rates = self.input_data["RateFileList"] 
        self.plasma_chemistry = Chemistry(path_to_rates) 
        initial_conditions = self.input_data["initial_conditions"]
        self.set_mobile_species()
        self.set_initial_plasma_quantities(initial_conditions)
        self.set_electron_species()

    def initialize(self):
        self.dt =  self.owner.clock.dt
                    
    def set_initial_plasma_quantities(self, initial_conditions):
        """
        This function determines the shape and size of the grid and sets the 
        initial conditions for each specties accordingly. 
        
        initial_conditions   - a dictionary of species that have non-zero partial pressures at t=0
        
        """
        NLoschmidt = 2.686E25                        # This is the number of molecules in an ideal gas at stp
        IC_keys = initial_conditions.keys()
        self.Fluid = {}
        for species in self.plasma_chemistry.species:
            mobile_species = (species in self.mobile)
            if species.name in IC_keys:
                ic = initial_conditions[species.name]
                pressure = ic['pressure']
                density = NLoschmidt*pressure
                velocity = ic['velocity']
                energy =   ic['energy']
                print("Pressure:",pressure)
                print("Velocity:",velocity)
                print("Energy:",energy)
                self.Fluid[species]=FluidSpecies(species, mobile_species, self.owner, density, velocity, energy)
            else:
                self.Fluid[species]=FluidSpecies(species, mobile_species, self.owner)
        return 
        
    def set_electron_species(self):
        for s in self.plasma_chemistry.species:
            if s.name == 'e':
                self.electron_species = s
        return

    def set_mobile_species(self):
        self.mobile = []
        for s in self.plasma_chemistry.species:
            if s.name == 'e':
                self.mobile.append(s)
        return

#     def predictor_corrector(self):
#         """
#         This function performs a predictor-corrector for updating the densities,
#         the electron velocity and energy
#         """
#         dt = self.owner.clock.dt
# #  This is the predictor step
#         sp_predictor = self.integrator(self.species, 0.5*dt)
# #  This is the corrector step
#         self.species = self.integrator(sp_predictor, dt)
# 
#         return
#         
#     def integrator(self, species, dt):
#         sp_copy = species.copy()
#         rhs = self.chem.RHS(sp_copy)
#         for s in species.keys():
#             sp_copy[s].density = species[s].density + rhs['density'][s] * dt
#         sp_copy['e'].velocity = species['e'].velocity - rhs['nu_m']['e']*species['e'].velocity* dt
#         sp_copy['e'].energy = species['e'].energy   + rhs['energy']['e'] * dt
#         return sp_copy
        
    def density_source(self, RX):

        rx_rate = self.get_reaction_rate( RX )
        source={}
        
        for reactant in RX.reactants:
            source[reactant] = 0
        for product in RX.products:
            source[product] =  0

        for reactant in RX.reactants:
            source[reactant] = source[reactant] - rx_rate
        for product in RX.products:
            source[product] =  source[product]  + rx_rate
        return source
                
    def get_reaction_rate(self, RX):
#  Get the rate coefficient
        k = (1.0E-6)*RX.get_rate_constant(self.energy)                           
# Determine the reaction rate from k
        density_product = 1
        for r in RX.reactants:
            nA = self.Fluid[r].density
            density_product = density_product*nA
        reaction_rate = density_product*k
        return reaction_rate
                
    def RHS(self):
        """  Add source terms to the Fluid for a specified set of reactions
        """
        self.energy = self.Fluid[self.electron_species].nEnergy/self.Fluid[self.electron_species].density
    #
    # Take care of changes to the density due to exciation collisions which include ionization, recombination, etc
        reactions = self.plasma_chemistry.excitation_reactions
        for RX in reactions:
            density_source_term = self.density_source( RX )
            for species in density_source_term:
                self.Fluid[species].density += density_source_term[species]*self.dt
    #
    #  Take care of momentum transfer
        reactions = self.plasma_chemistry.momentum_transfer_reactions
        for RX in reactions:
            momentum_Xfer = self.get_reaction_rate( RX )/self.Fluid[self.electron_species].density
            self.Fluid[self.electron_species].nV += - self.Fluid[self.electron_species].nV*momentum_Xfer*self.dt
            # mu = self.echarge/(9.11e-31*momentum_Xfer)
            # self.Fluid[self.electron_species].nV[:] = -self.Fluid[self.electron_species].density*mu*self.E
            
    #
    #  Take care of energy losses due to inelastic collisions
        reactions = self.plasma_chemistry.excitation_reactions
        for RX in reactions:
            rx_rate = self.get_reaction_rate( RX )
            deltaE = RX.delta_e
            self.Fluid[self.electron_species].nEnergy += -rx_rate*deltaE*self.dt
            
    def fluid_pusher(self):
        for s in self.mobile:
            EM_forces = s.charge/s.mass*self.E
            self.Fluid[s].nV += self.Fluid[s].density*EM_forces*self.dt

    def OhmicHeating(self):
        for s in self.mobile:
            Z = np.sign(s.charge/self.echarge)
#            self.Fluid[s].nEnergy += Z*self.Fluid[s].velocity*self.E.T
            self.Fluid[s].nEnergy += Z*self.Fluid[s].nV*self.E*self.dt

    def forward_Euler(self):
        '''  This function first takes care of the plasma chemistry and 
        then does the terms specific to mobile charged particles.  The call to 
        fluid_pusher does the Lorentz force update and the call to OhmicHeating 
        adds the J.E term to the energy equation.
        '''
        self.RHS()
        self.fluid_pusher() 
        self.OhmicHeating()
        self.updateJ()

    def updateJ(self):
        self.J = 0
#  No return current feedback to the field solver yet.  
        return
#
        for s in self.mobile:
            self.J += s.charge*self.Fluid[s].nV
        return

    def exchange_resources(self):
        self.publish_resource({"ResponseModel:J": self.J})
        for s in self.Fluid:
            self.publish_resource({"FluidModel:" + s.name:self.Fluid[s]})
#        self.publish_resource({"GasDensity": self.species['N2(X1)'].density})
#        self.publish_resource({"MomentumTransfer": self.nu_m})
        
    def inspect_resource(self, resource):
        if "FieldModel:E" in resource:
            print("adding E-field resource")
            self.E = resource["FieldModel:E"]
        if "FieldModel:B" in resource:
            print("adding B-field resource")
            self.B = resource["FieldModel:B"]

class FluidDiagnostic(Diagnostic):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        
        self.component = input_data["component"]
        self.fluid_name = input_data["fluid_name"]
        self.output = input_data["output"] 
        self.field = None
        self.units = "Diagnostic Units"
        self.file = None
        
    def diagnose(self):
        if self.component=='energy':
            self.output_function(self.fluid.nEnergy/self.fluid.density)
            self.units = "(eV)"
        elif self.component=='Vz':
            self.output_function(self.fluid.nV/self.fluid.density*1E-7)
            self.units = "(cm/ns)"
        elif self.component=='density':      
            self.output_function(self.fluid.__dict__[self.component]*1E-6)
            self.units = "cm^-3"
        else:
            print("Warning:  Fluid diagnostic component undefined")
            

    def inspect_resource(self, resource):
        if self.fluid_name in resource:
            self.fluid = resource[self.fluid_name]
    
    def print_diag(self, data):
        print(self.fluid_name, self.component,self.units, data)
        
    def initialize(self):
        # setup output method
        functions = {"stdout": self.print_diag,
                     "csv": self.write_to_csv,
                     }
        self.output_function = functions[self.input_data["output"]]
        if self.input_data["output"] == "csv":
            self.outputbuffer = np.zeros((
                        self.owner.clock.num_steps+1,
                        self.owner.grid.num_points
                        ))
    
    def write_to_csv(self, data):
        i = self.owner.clock.this_step
        self.outputbuffer[i,:] = data[:,0]
    
    def finalize(self):
        self.diagnose()
        if self.input_data["output"] == "csv":
            self.file = open(self.input_data["filename"], 'wb')
            np.savetxt(self.file, self.outputbuffer, delimiter=",")
            self.file.close()

Diagnostic.add_diagnostic_to_library("fluid",FluidDiagnostic)


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
        self.J[:] = (time<2*self.rise_time)*np.sin(np.pi*time/self.rise_time/2)**2 * self.profile        

Module.add_module_to_library("FieldModel", FieldModel)
Module.add_module_to_library("ThermalFluidPlasma", ThermalFluidPlasma)
Module.add_module_to_library("RigidBeamCurrentSource", RigidBeamCurrentSource)
##
#  Chemistry files
# p = Path('chemistry/N2_Rates_TT_wo_recombination.txt')
p = Path('chemistry/N2_Rates_TT.txt')
RateFileList=[str(p)]
# Initial Conditions:
#       All species that do not have a non-zero partial pressure at t=0 will be set to zero
#

initial_conditions={     'e':{'pressure':1e-5*1.0/760,'velocity':0.0,'energy':0.1},
                     'N2(X1)':{'pressure':1.0/760,'velocity':0.0,'energy':1.0/40.0}
                     }                    

end_time = 150.E-9
dt = 0.1E-9
number_of_steps = int(end_time/dt)
#number_of_steps = 200
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
        # {"type": "fluid",
        #      "fluid_name": "FluidModel:e",
        #      "output": "stdout",
        #      "component": "density",
        #  },
        {"type": "fluid",
             "fluid_name": "FluidModel:e",
             "output": "csv",
             "filename": "Energy.csv",
             "component": "energy",
         },
        # {"type": "fluid",
        #      "fluid_name": "FluidModel:e",
        #      "output": "stdout",
        #      "component": "Vz",
        #  },
        # {"type": "fluid",
        #      "fluid_name": "FluidModel:N2(X1)",
        #      "output": "stdout",
        #      "component": "density",
        #  },
        # {"type": "fluid",
        #      "fluid_name": "FluidModel:N2(Rot)",
        #      "output": "stdout",
        #      "component": "density",
        #  },
        # {"type": "fluid",
        #      "fluid_name": "FluidModel:N2(v1)",
        #      "output": "stdout",
        #      "component": "density",
        # },
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