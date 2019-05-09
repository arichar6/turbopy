from turbopy import Simulation, Module, Diagnostic, CSVDiagnosticOutput
import numpy as np
import chemistry as ch
# from chemistry import Species, Chemistry, Reaction, electron_species, N2_ground_state

import scipy.special as special      # for i1, k1 modified Bessel functions
import scipy.integrate as integrate
from scipy import sparse

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
        # Plasma current and E are inferred from the solution
        self.J_plasma = owner.grid.generate_field(1)
        self.E = owner.grid.generate_field(1)
        
        self.source = owner.grid.generate_field(1)
        
        # Externally driven currents act a sources in the EMHD equation
        self.dJdr = None
        self.J_beam = None
        
        self.electron_fluid = None
        
        self.dr = owner.grid.dr
        
        self.rad_diff = None
                
    def initialize(self):
        self.dt = self.owner.clock.dt
        if ch.electron_species in self.FluidModel:
            self.electron_fluid = self.FluidModel[ch.electron_species]
        else:
            raise RuntimeError("Electron fluid species not found")
        
        self.construct_rad_diff()
    
    def inspect_resource(self, resource):
        if "CurrentSource:dJdr" in resource:
            self.dJdr = resource["CurrentSource:dJdr"]
        if "CurrentSource:J" in resource:
            self.J_beam = resource["CurrentSource:J"]
        if "FluidModel" in resource:
            self.FluidModel = resource["FluidModel"]
                                
    def exchange_resources(self):
        self.publish_resource({"EMHD:Omega": self.Omega})
        self.publish_resource({"EMHD:B": self.B})
        self.publish_resource({"EMHD:E": self.E})
        self.publish_resource({"EMHD:source": self.source})
        self.publish_resource({"EMHD:J_plasma": self.J_plasma})

    def update(self):
        m = ch.electron_species.mass * 1.6605402E-27
        self.alpha = ((self.mu0 * self.electron_fluid.density * ch.echarge**2) / m)**0.5
        
        self.invert_omega()
        self.update_omega()
        self.update_current()
    
    def invert_omega(self):
        # B = I1 * int_R(K1 * source) + K1 * int_L(I1 * source)
        
        # alpha changes because density can change? Is this ok?
        i1_grid = special.i1(self.owner.grid.r * self.alpha)
        k1_grid = special.k1(self.owner.grid.r * self.alpha)
        k1_grid[0] = 0  # artificially remove singularity at origin

        self.source[:] = self.Omega + self.mu0 * self.dJdr
        I1 = i1_grid * integrate.cumtrapz((k1_grid * self.source)[::-1], dx=self.dr, initial=0)[::-1]
        K1 = k1_grid * integrate.cumtrapz((i1_grid * self.source), dx=self.dr, initial=0)
        
        self.B[:] = I1 + K1
        self.B -= self.B[0]
        
    def update_omega(self):
        nu = 1
        self.Omega[:] += -nu * (self.Omega + self.alpha * self.B) * self.dt
    
    def update_current(self):
        # compute the plasma current from the magnetic field
        # Jp = (1/mu0) (1/r)(d/dr)(rB) - Jb
        # Use properties of modified Bessel functions?
        self.J_plasma[:] = -self.J_beam + (1/self.mu0) * (self.rad_diff @ self.B)

    def construct_rad_diff(self):
        # FD matrix for (rB)'/r = (1/r)(d/dr)(rB)
        N = self.owner.grid.num_points
        g = 1/(2.0 * self.dr)
        col_below = np.zeros(N)
        col_diag = np.zeros(N)
        col_above = np.zeros(N)
        col_below[:-1] = -g * (self.owner.grid.r[:-1]/self.owner.grid.r[1:])
        col_above[1:] = g * (self.owner.grid.r[1:]/self.owner.grid.r[:-1])
        # set boundary conditions
        # At r=0, use B~linear, and B=0.
        col_above[1] = 2.0 / self.dr
        # At r=Rw, use rB~const?
        col_diag[-1] = 1.0 / self.dr
        col_below[-2] = 2.0 * col_below[-1]
        # set main columns for finite difference derivative
        D = sparse.dia_matrix( ([col_below, col_diag, col_above], [-1, 0, 1]), shape=(N, N) )
        self.rad_diff = D



class ConductivityModel(Module):
    """Solve the coupled field and electron momentum equations"""
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.mu0 = 4 * np.pi * 1e-7
        m = ch.electron_species.mass * 1.6605402E-27
        self.mu0e2om = self.mu0 * ch.echarge**2 / m
        
        self.B = owner.grid.generate_field(1)
        self.E = owner.grid.generate_field(1)
        self.J_plasma = owner.grid.generate_field(1)
        self.dJp_source = owner.grid.generate_field(1)

        self.get_rate = None
        self.plasma_chemistry = None
        self.solver_name = input_data["solver"]
        self.solver = None

    def initialize(self):
        self.solver = self.owner.find_tool_by_name(self.solver_name)        
        self.D2 = self.solver.del2_radial()

        self.dt = self.owner.clock.dt
        if ch.electron_species in self.FluidModel:
            self.electron_fluid = self.FluidModel[ch.electron_species]
        else:
            raise RuntimeError("Electron fluid species not found")

    def inspect_resource(self, resource):
        if "CurrentSource:dJdt" in resource:
            self.dJdt = resource["CurrentSource:dJdt"]
        if "CurrentSource:J" in resource:
            self.J_beam = resource["CurrentSource:J"]
        if "FluidModel" in resource:
            self.FluidModel = resource["FluidModel"]
        if "PlasmaChemistry" in resource:
            self.plasma_chemistry = resource["PlasmaChemistry"]
        if "PlasmaChemistry:rate_function" in resource:
            self.get_rate = resource["PlasmaChemistry:rate_function"]
                                            
    def exchange_resources(self):
        self.publish_resource({"Fields:B": self.B})
        self.publish_resource({"Fields:E": self.E})
        self.publish_resource({"Fields:J_plasma": self.J_plasma})
        self.publish_resource({"Fields:dJp_source": self.dJp_source})
    
    def update(self):
        energy = self.electron_fluid.nEnergy / self.electron_fluid.density 
        nu = 0
        for RX in self.plasma_chemistry.momentum_transfer_reactions:
            nu += self.get_rate(RX, energy)
        nu = nu / self.electron_fluid.density
        mu0sigma = self.mu0e2om * self.electron_fluid.density / nu
        
        self.dJp_source[:] = (1/mu0sigma) * (self.D2 @ self.J_plasma)
        self.J_plasma[:] += ((1/mu0sigma) * (self.D2 @ self.J_plasma) - self.dJdt) * self.dt
        
        self.B[:] = np.cumsum(self.owner.grid.r * (self.J_plasma + self.J_beam))
        self.B[:] *= self.owner.grid.dr / self.owner.grid.r
        self.B[0] = 0
        
        self.E[:] = self.mu0 * self.J_plasma / mu0sigma
        


class FluidSpecies(ch.Species):
    def __init__(self, species, owner, density=0):
        self.generate_field = owner.grid.generate_field
        super().__init__(species.mass, species.charge, species.name)
        self.density = density * (1 + self.generate_field(1))
        
    def mobilize(self, velocity=0, energy=0):
        self.nV = self.density * velocity * (1 + self.generate_field(1))
        self.nEnergy = self.density * energy * (1 + self.generate_field(1))



class PlasmaChemistry(Module):
    """
    This is where the reaction rates are read from file
    list_of_rate_files   - is the path to the chemistry file containing rate coefficients
    chem_dir             - is the path directory where gas_chemistry.py resides
    """
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        path_to_rates = self.input_data["RateFileList"] 
        self.plasma_chemistry = ch.Chemistry(path_to_rates)
        
        self.Fluid = None

        # Do something else here to set "weakly ionized" approximation?

    def exchange_resources(self):
        self.publish_resource({"PlasmaChemistry": self.plasma_chemistry})
        self.publish_resource({"PlasmaChemistry:rate_function": self.get_reaction_rate})
    
    def inspect_resource(self, resource):
        if "FluidModel" in resource:
            self.Fluid = resource["FluidModel"]
        
    def get_reaction_rate(self, RX, energy):
        #  Get the rate coefficient
        k = (1.0E-6) * RX.get_rate_constant(energy)                           
        # Determine the reaction rate from k
        density_product = 1
        for r in RX.reactants:
            nA = self.Fluid[r].density
            density_product = density_product * nA
        reaction_rate = density_product * k
        return reaction_rate
        
    def update(self):
        pass



class FluidElectrons(Module):
    """
    This is the module that takes care of updating the fluid momentum and energy equations
    """
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.plasma_chemistry = None
        self.electron_fluid = None
        self.J_plasma = None
        
        self.get_rate = None
    
    def inspect_resource(self, resource):
        if "PlasmaChemistry" in resource:
            self.plasma_chemistry = resource["PlasmaChemistry"]
        if "FluidModel" in resource:
            self.FluidModel = resource["FluidModel"]
        if "Fields:J_plasma" in resource:
            self.J_plasma = resource["Fields:J_plasma"]
        if "Fields:E" in resource:
            self.E = resource["Fields:E"]
        if "PlasmaChemistry:rate_function" in resource:
            self.get_rate = resource["PlasmaChemistry:rate_function"]
    
    def initialize(self):
        self.dt = self.owner.clock.dt
        
        if ch.electron_species in self.FluidModel:
            self.electron_fluid = self.FluidModel[ch.electron_species]
        else:
            raise RuntimeError("Electron fluid species not found")

        ic = self.input_data["initial_conditions"]["e"]
        self.electron_fluid.mobilize(ic["velocity"], ic["energy"])
            
    def update_energy(self):
        #  Take care of energy losses due to inelastic collisions
        energy = self.electron_fluid.nEnergy / self.electron_fluid.density
        for RX in self.plasma_chemistry.excitation_reactions:
            rx_rate = self.get_rate(RX, energy)
            deltaE = RX.delta_e
            self.electron_fluid.nEnergy += -rx_rate * deltaE * self.dt
        
        # Update energy based on Ohmic heating
        # NB factor of electron charge is not here because units of energy are eV
        self.electron_fluid.nEnergy += - self.electron_fluid.nV * self.E * self.dt        

        # What about changes in energy due to elastic scattering in COM frame, between 
        # species with very different mass?

    def update_momentum(self):
        self.electron_fluid.nV = self.J_plasma / (-ch.echarge * self.electron_fluid.density)
    
    def update(self):
        # note that the momentum should be updated based on the solution from EMHD
        self.update_momentum()

        # this is where the electron energy gets updated
        self.update_energy()


                
class ThermalFluidPlasma(Module):
    """
    This class handles the themal plasma (fluid-like) response that is used to 
    compute plasma return currents.  
    """
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.J = self.owner.grid.generate_field(1)  # only Jz for now
        self.E = None

        self.update = self.forward_Euler
        self.plasma_chemistry = None
        self.Fluid = {}
        self.get_rate = None
        
    def initialize(self):
        self.dt =  self.owner.clock.dt
        initial_conditions = self.input_data["initial_conditions"]
        self.set_initial_plasma_quantities(initial_conditions)

    def exchange_resources(self):
        self.publish_resource({"FluidModel": self.Fluid})
        
    def inspect_resource(self, resource):
        if "PlasmaChemistry" in resource:
            self.plasma_chemistry = resource["PlasmaChemistry"]
        if "PlasmaChemistry:rate_function" in resource:
            self.get_rate = resource["PlasmaChemistry:rate_function"]
            
    def set_initial_plasma_quantities(self, initial_conditions):
        """
        This function determines the shape and size of the grid and sets the 
        initial conditions for each specties accordingly. 
        
        initial_conditions   - a dictionary of species that have non-zero partial pressures at t=0
        """
        NLoschmidt = 2.686E25                        # This is the number of molecules in an ideal gas at stp
        IC_keys = initial_conditions.keys()
        for species in self.plasma_chemistry.species:
            if species.name in IC_keys:
                ic = initial_conditions[species.name]
                pressure = ic['pressure']
                density = NLoschmidt * pressure
                velocity = ic['velocity']
                energy = ic['energy']
                print("Pressure:", pressure)
                print("Velocity:", velocity)
                print("Energy:", energy)
                self.Fluid[species] = FluidSpecies(species, self.owner, density)
            else:
                self.Fluid[species] = FluidSpecies(species, self.owner)
        return 
        
    def density_source(self, RX, energy):
        rx_rate = self.get_rate(RX, energy)
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
    
    def update_densities(self, energy):
        # Take care of changes to the density due to exciation collisions which include ionization, recombination, etc
        for RX in self.plasma_chemistry.excitation_reactions:
            density_source_term = self.density_source(RX, energy)
            for species in density_source_term:
                self.Fluid[species].density += density_source_term[species]*self.dt

    def forward_Euler(self):
        energy = self.Fluid[ch.electron_species].nEnergy/self.Fluid[ch.electron_species].density
        self.update_densities(energy)



class RigidBeamCurrentSource(Module):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.input_data = input_data
        self.peak_current = input_data["peak_current"]
        self.beam_radius = input_data["beam_radius"]
        self.rise_time = input_data["rise_time"]

        self.J = owner.grid.generate_field(1)           # only Jz for now
        self.dJdr = owner.grid.generate_field(1)
        self.dJdt = owner.grid.generate_field(1)
        self.profile = owner.grid.generate_field(1)
        self.ddr_profile = owner.grid.generate_field(1)
        self.set_profile(input_data["profile"])
        self.set_ddr_profile(input_data["profile"])

    def initialize(self):
        self.dt =  self.owner.clock.dt
                        
    def exchange_resources(self):
        self.publish_resource({"CurrentSource:J": self.J})
        self.publish_resource({"CurrentSource:dJdr": self.dJdr})
        self.publish_resource({"CurrentSource:dJdt": self.dJdt})
    
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
    
    def set_ddr_profile(self, profile_type):
        profiles = {"gaussian": lambda r: 
                        -2 * (r/self.beam_radius**2) * self.peak_current * np.exp(-(r/self.beam_radius)**2),
                    }
        try:
            self.ddr_profile[:] = profiles[profile_type](self.owner.grid.r)
        except KeyError:
            raise KeyError("Unknown profile type: {0}".format(profile_type))        
            
    def update(self):
        self.set_current_for_time(self.owner.clock.time)
    
    def set_current_for_time(self, time):
        self.dJdt[:] = self.J[:]
        self.J[:] = (time<2*self.rise_time)*np.sin(np.pi*time/self.rise_time/2)**2 * self.profile
        self.dJdr[:] = (time<2*self.rise_time)*np.sin(np.pi*time/self.rise_time/2)**2 * self.ddr_profile
        self.dJdt[:] -= self.J[:]
        self.dJdt[:] = -self.dJdt[:]/self.dt
        


Module.register("ConductivityModel", ConductivityModel)
Module.register("PlasmaChemistry", PlasmaChemistry)
Module.register("EMHD", EMHD)
Module.register("ThermalFluidPlasma", ThermalFluidPlasma)
Module.register("RigidBeamCurrentSource", RigidBeamCurrentSource)
Module.register("FluidElectrons", FluidElectrons)



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
        s_key = ch.Species(name=self.fluid_name,charge=0,mass=0)
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

end_time = 30E-9
dt = 0.1E-9
number_of_steps = int(end_time/dt)
#number_of_steps = 200
N_grid = 50

sim_config = {"Modules": [
        {"name": "PlasmaChemistry",
            "RateFileList": RateFileList
        },
        {"name": "RigidBeamCurrentSource",
             "peak_current": 1.0e5,
             "beam_radius": 0.02,
             "rise_time": 30.0e-9,
             "profile": "gaussian",
        },
        {"name": "ThermalFluidPlasma",
            "initial_conditions": initial_conditions
        },
        {"name": "FluidElectrons",
            "initial_conditions": initial_conditions
        },
        {"name": "ConductivityModel", 
            "solver": "FiniteDifference"
        },
    ],
    "Diagnostics": [
        {"type": "field",
             "field": "Fields:E",
             "output": "csv",
             "filename": "beam_output/E.csv",
             "component": 0,
         },
        {"type": "field",
             "field": "Fields:B",
             "output": "csv",
             "filename": "beam_output/B.csv",
             "component": 0,
         },
         {"type": "field",
             "field": "CurrentSource:J",
             "output": "csv",
             "filename": "beam_output/BeamCurrent.csv",
             "component": 0,
         },
         {"type": "field",
             "field": "CurrentSource:dJdr",
             "output": "csv",
             "filename": "beam_output/BeamCurrentGradient.csv",
             "component": 0,
         },
         {"type": "field",
             "field": "Fields:dJp_source",
             "output": "csv",
             "filename": "beam_output/source.csv",
             "component": 0,
         },
         {"type": "field",
             "field": "Fields:J_plasma",
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
        {"type": "FiniteDifference",
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