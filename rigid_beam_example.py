from turbopy import Simulation, Module, Diagnostic, CSVDiagnosticOutput
import numpy as np
from chemistry import Species, Chemistry

from pathlib import Path


class EMHD(Module):
    """
    Solve EMHD equations for the RB problem.
    This module uses some approximations to the field equations,
    together with a fluid model for the plasma electrons.
    """
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.mu0 = 4 * np.pi * 1e-7
        
        # EMHD solves for B and Omega as functions of time
        self.B = owner.grid.generate_field(1)
        self.Omega = owner.grid.generate_field(1)
        
        # Externally driven currents act a sources in the EMHD equation
        self.currents = []
        
        # Use a separate solver tool to do the integrals
        self.solver_name = input_data["solver"]
        self.solver = None

    def initialize(self):
        self.solver = self.owner.find_tool_by_name(self.solver_name)
    
    def inspect_resource(self, resource):
        if "CurrentSource:dJdr" in resource:
            self.currents.append(resource["CurrentSource:dJdr"])
        if "FluidModel" in resource:
            for k, v in resource["FluidModel"]
                if k.name == 'e':
                    self.electron_density = v
            
    def exchange_resources(self):
        """Tell other modules about the electric field, in case the need it"""
        self.publish_resource({"EMHD:Omega": self.Omega})
        self.publish_resource({"EMHD:B": self.B})

    def update(self):
        # return
        self.old_source[:] = self.sourceterm[:]
        self.sourceterm[:] = np.sum(self.currents, axis=0)
        dt = self.owner.clock.dt
        dJ = (self.sourceterm - self.old_source)/dt
        self.E[:] = self.solver.solve( self.mu0 * dJ[:] )
        
        

class FieldModel(Module):
    """
    This is the rigid beam field module
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
        self.E[:] = self.solver.solve( self.mu0 * dJ[:] )



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



class PlasmaChemistry(Module):
    """
    This is where the plasma chemistry is handled
    list_of_rate_files   - is the path to the chemistry file containing rate coefficients
    chem_dir             - is the path directory where gas_chemistry.py resides
    """
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        path_to_rates = self.input_data["RateFileList"] 
        self.plasma_chemistry = Chemistry(path_to_rates) 

    def exchange_resources(self):
        self.publish_resource({"PlasmaChemistry": self.plasma_chemistry})
        
    def update(self):
        pass


                
class ThermalFluidPlasma(Module):
    """
    This class handles the themal plasma (fluid-like) response that is used to 
    compute plasma return currents.  The only required function is update_J which
    provides the plasma return current.
    q                    - is the species charge
    m                    - is the species mass
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
        self.plasma_chemistry = None
        self.Fluid = {}
                
    def initialize(self):
        self.dt =  self.owner.clock.dt
        initial_conditions = self.input_data["initial_conditions"]
        self.set_mobile_species()
        self.set_initial_plasma_quantities(initial_conditions)
        self.set_electron_species()
                    
    def set_initial_plasma_quantities(self, initial_conditions):
        """
        This function determines the shape and size of the grid and sets the 
        initial conditions for each specties accordingly. 
        
        initial_conditions   - a dictionary of species that have non-zero partial pressures at t=0
        
        """
        NLoschmidt = 2.686E25                        # This is the number of molecules in an ideal gas at stp
        IC_keys = initial_conditions.keys()
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
        
    def density_source(self, RX, energy):

        rx_rate = self.get_reaction_rate(RX, energy)
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
                
    def get_reaction_rate(self, RX, energy):
        #  Get the rate coefficient
        k = (1.0E-6)*RX.get_rate_constant(energy)                           
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
        energy = self.Fluid[self.electron_species].nEnergy/self.Fluid[self.electron_species].density
        #
        # Take care of changes to the density due to exciation collisions which include ionization, recombination, etc
        for RX in self.plasma_chemistry.excitation_reactions:
            density_source_term = self.density_source(RX, energy)
            for species in density_source_term:
                self.Fluid[species].density += density_source_term[species]*self.dt
        #
        #  Take care of momentum transfer
        for RX in self.plasma_chemistry.momentum_transfer_reactions:
            momentum_Xfer = self.get_reaction_rate(RX, energy)/self.Fluid[self.electron_species].density
            self.Fluid[self.electron_species].nV += - self.Fluid[self.electron_species].nV*momentum_Xfer*self.dt
            # mu = self.echarge/(9.11e-31*momentum_Xfer)
            # self.Fluid[self.electron_species].nV[:] = -self.Fluid[self.electron_species].density*mu*self.E
            
        #
        #  Take care of energy losses due to inelastic collisions
        for RX in self.plasma_chemistry.excitation_reactions:
            rx_rate = self.get_reaction_rate(RX, energy)
            deltaE = RX.delta_e
            self.Fluid[self.electron_species].nEnergy += -rx_rate*deltaE*self.dt
        
        # What about changes in energy due to elastic scattering in COM frame, between 
        # species with very different mass?
        
    def fluid_pusher(self):
        for s in self.mobile:
            EM_forces = -s.charge/s.mass*self.E
            self.Fluid[s].nV += self.Fluid[s].density*EM_forces*self.dt

    def OhmicHeating(self):
        for s in self.mobile:
            Z = np.sign(s.charge/self.echarge)
            # where is the factor of q_e, the electron charge?
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
        self.J[:] = 0

        #  No return current feedback to the field solver yet.  
        # return
        #
        for s in self.mobile:
            self.J += s.charge*self.Fluid[s].nV
        return

    def exchange_resources(self):
        self.publish_resource({"ResponseModel:J": self.J})
        self.publish_resource({"FluidModel": self.Fluid})
#         for s in self.Fluid:
#             self.publish_resource({"FluidModel:" + s.name: self.Fluid[s]})
        
    def inspect_resource(self, resource):
        if "FieldModel:E" in resource:
            print("adding E-field resource")
            self.E = resource["FieldModel:E"]
        if "FieldModel:B" in resource:
            print("adding B-field resource")
            self.B = resource["FieldModel:B"]
        if "PlasmaChemistry" in resource:
            self.plasma_chemistry = resource["PlasmaChemistry"]



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
            self.profile[:] = profiles[profile_type](self.owner.grid.r)
        except KeyError:
            raise KeyError("Unknown profile type: {0}".format(profile_type))
            
    def update(self):
        self.set_current_for_time(self.owner.clock.time)
    
    def set_current_for_time(self, time):
        self.J[:] = (time<2*self.rise_time)*np.sin(np.pi*time/self.rise_time/2)**2 * self.profile        


Module.register("PlasmaChemistry", PlasmaChemistry)
Module.register("FieldModel", FieldModel)
Module.register("ThermalFluidPlasma", ThermalFluidPlasma)
Module.register("RigidBeamCurrentSource", RigidBeamCurrentSource)



class FluidDiagnostic(Diagnostic):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        
        self.component = input_data["component"]
        self.fluid_name = input_data["fluid_name"]
        self.output = input_data["output"] 
        self.field = None
        self.units = "Diagnostic Units"
        
    def diagnose(self):
        if self.component=='energy':
            self.output_function(self.fluid.nEnergy/self.fluid.density)
            self.units = "(eV)"
        elif self.component=='Vz':
            self.output_function(self.fluid.nV/self.fluid.density * 1E-7)
            self.units = "(cm/ns)"
        elif self.component=='density':      
            self.output_function(self.fluid.density * 1E-6)
            self.units = "cm^-3"
        else:
            # print("Warning:  Fluid diagnostic component undefined")
            self.output_function(self.fluid.__dict__[self.component])

    def inspect_resource(self, resource):
        if "FluidModel" in resource:
            self.fluid_model = resource["FluidModel"]
    
    def print_diagnose(self, data):
        print(self.fluid_name, self.component,self.units, data)
        
    def initialize(self):
        # set up fluid
        s_key = Species(name=self.fluid_name,charge=0,mass=0)
        if s_key in self.fluid_model:
            self.fluid = self.fluid_model[s_key]
        else:
            raise RuntimeError("Fluid "+self.fluid_name+" not found in model")
            
        # setup output method
        functions = {"stdout": self.print_diagnose,
                     "csv": self.csv_diagnose,
                     }
        self.output_function = functions[self.input_data["output"]]
        if self.input_data["output"] == "csv":
            diagnostic_size = (self.owner.clock.num_steps+1,
                               self.owner.grid.num_points)
            self.csv = CSVDiagnosticOutput(self.input_data["filename"], diagnostic_size)
    
    def csv_diagnose(self, data):
        self.csv.append(data)
    
    def finalize(self):
        self.diagnose()
        if self.input_data["output"] == "csv":
            self.csv.finalize()

            
Diagnostic.register("fluid", FluidDiagnostic)



#  Chemistry files
# p = Path('chemistry/N2_Rates_TT_wo_recombination.txt')
p = Path('chemistry/N2_Rates_TT.txt')
RateFileList=[str(p)]
# Initial Conditions:
#       All species that do not have a non-zero partial pressure at t=0 will be set to zero
#

initial_conditions={      'e':{'pressure':1e-5*1.0/760,'velocity':1e-6,'energy':0.1},
                     'N2(X1)':{'pressure':1.0/760,'velocity':0.0,'energy':1.0/40.0}
                     }                    

end_time = 1.E-10
dt = 1E-11
number_of_steps = int(end_time/dt)
#number_of_steps = 200
N_grid = 8

sim_config = {"Modules": [
        {"name": "FieldModel",
             "solver": "PoissonSolver1DRadial",
         },
        {"name": "PlasmaChemistry",
        "RateFileList":RateFileList
        },
        {"name": "ThermalFluidPlasma",
        "initial_conditions":initial_conditions
         },
        {"name": "RigidBeamCurrentSource",
             "peak_current": 0, # 1.0e5,
             "beam_radius": 0.05,
             "rise_time": 30.0e-9,
             "profile": "uniform",
         },
    ],
    "Diagnostics": [
        {"type": "field",
             "field": "FieldModel:E",
             "output": "csv",
             "filename": "beam_output/Efield.csv",
             "component": 0,
         },
        {"type": "field",
             "field": "CurrentSource:J",
             "output": "csv",
             "filename": "beam_output/BeamCurrent.csv",
             "component": 0,
         },
        {"type": "field",
             "field": "ResponseModel:J",
             "output": "csv",
             "filename": "beam_output/PlasmaCurrent.csv",
             "component": 0,
         },
        {"type": "fluid",
             "fluid_name": "e",
             "output": "csv",
             "filename": "beam_output/ElectronDensity.csv",
             "component": "density",
         },
        {"type": "fluid",
             "fluid_name": "e",
             "output": "csv",
             "filename": "beam_output/Energy.csv",
             "component": "energy",
         },
        {"type": "fluid",
             "fluid_name": "e",
             "output": "csv",
             "filename": "beam_output/electron_Vz.csv",
             "component": "Vz",
         },
        {"type": "fluid",
             "fluid_name": "e",
             "output": "csv",
             "filename": "beam_output/electron_nV.csv",
             "component": "nV",
         },
        {"type": "fluid",
             "fluid_name": "N2(X2:ion)",
             "output": "csv",
             "filename": "beam_output/N2(X2:ion)_Density.csv",
             "component": "density",
         },
        {"type": "fluid",
             "fluid_name": "N2(X1)",
             "output": "csv",
             "filename": "beam_output/N2(X1)_Density.csv",
             "component": "density",
         },
        {"type": "fluid",
             "fluid_name": "N2(Rot)",
             "output": "csv",
             "filename": "beam_output/N2(Rot)_Density.csv",
             "component": "density",
         },
        {"type": "fluid",
             "fluid_name": "N2(v1)",
             "output": "csv",
             "filename": "beam_output/N2(v1)_Density.csv",
             "component": "density",
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