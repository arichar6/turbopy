# Computational Physics Simulation Framework
#
# Based on the structure of turboWAVE
#
import numpy as np

class Simulation:
    """
    This "owns" all the physics modules and compute tools, and coordinates them.
    The main simulation loop is driven by an instance of this class.
    
    Based on the Simulation class in TurboWAVE
    """
    def __init__(self, input_data: dict):
        self.modules = []
        self.compute_tools = []
        self.diagnostics = []

        self.grid = None
        self.clock = None
        self.units = None
        
        self.input_data = input_data
    
    def run(self):
        print("Simulation is initializing")
        self.prepare_simulation()
        print("Initialization complete")
        
        print("Simulation is started")
        while self.clock.is_running():
            self.fundamental_cycle()
        
        self.finalize_simulation()        
        print("Simulation complete")
    
    def fundamental_cycle(self):
        for d in self.diagnostics:
            d.diagnose()
        for m in self.modules:
            m.reset()
        for m in self.modules:
            m.update()
        self.clock.advance()

    def prepare_simulation(self):
        print("Reading Grid...")
        self.read_grid_from_input()
        
        print("Reading Tools...")
        self.read_tools_from_input()
        
        print("Reading Modules...")
        self.read_modules_from_input()
        self.sort_modules()
        
        print("Reading Diagnostics...")
        self.read_diagnostics_from_input()
        
        print("Initializing Simulation Clock...")
        self.read_clock_from_input()
        
        print("Initializing Tools...")
        for t in self.compute_tools:
            t.initialize()
        
        print("Initializing Modules...")
        for m in self.modules:
            m.exchange_resources()
        for m in self.modules:
            m.initialize()

        print("Initializing Diagnostics...")
        for d in self.diagnostics:
            d.initialize()
            
    def finalize_simulation(self):
        for d in self.diagnostics:
            d.finalize()
    
    def read_grid_from_input(self):
        self.grid = Grid(self.input_data["Grid"])
    
    def read_clock_from_input(self):
        self.clock = SimulationClock(self, self.input_data["Clock"])
    
    def read_tools_from_input(self):
        pass
    
    def read_modules_from_input(self):
        for module_data in self.input_data["Modules"]:
            module_class = Module.lookup_name(module_data["name"])
            self.modules.append(module_class(owner=self, input_data=module_data))
    
    def read_diagnostics_from_input(self):
        for d in self.input_data["Diagnostics"]:
            diagnostic_class = Diagnostic.lookup_name(d["type"])
            self.diagnostics.append(diagnostic_class(owner=self, input_data=d))
    
    def sort_modules(self):
        pass
            

class Module:
    """
    This is the base class for all physics modules
    Based on Module class in TurboWAVE

    Because python mutable/immutable is different than C++ pointers, the implementation 
    here is different. Here, a "resource" is a dictionary, and can have more than one 
    thing being shared. Note that the value stored in the dictionary needs to be mutable. 
    Make sure not to reinitialize, because other modules will be holding a reference to it.
    """
    module_library = {}
    
    @classmethod
    def add_module_to_library(cls, module_name, module_class):
        if module_name in cls.module_library:
            raise ValueError("Module '{0}' already in module library".format(module_name))
        cls.module_library[module_name] = module_class
    
    @classmethod
    def lookup_name(cls, module_name):
        try:
            return cls.module_library[module_name]
        except KeyError:
            raise KeyError("Module '{0}' not found in module library".format(module_name))
    
    def __init__(self, owner: Simulation, input_data: dict):
        self.owner = owner
        self.module_type = None
        self.input_data = input_data

    def publish_resource(self, resource: dict):
        for module in self.owner.modules:
            module.inspect_resource(resource)
        for diagnostic in self.owner.diagnostics:
            diagnostic.inspect_resource(resource)

    def inspect_resource(self, resource: dict):
        """
        If your subclass needs the data described by the key, now's their chance to 
        save a pointer to the data
        """
        pass

    def exchange_resources(self):
        """
        This is the function where you call publish_resource, to tell other modules 
        about data you want to share
        """
        pass

    def update(self):
        raise NotImplementedError
    
    def reset(self):
        pass
    
    def initialize(self):
        pass
        

class ComputeTool:
    """
    This is the base class for compute tools. These are the compute-heavy functions,
    which have implementations of numerical methods which can be shared between modules.
    """
    def __init__(self, owner: Simulation):
        self.owner = owner
        raise NotImplementedError


class SimulationClock:
    def __init__(self, owner: Simulation, clock_data: dict):
        self.owner = owner
        self.start_time = clock_data["start_time"]
        self.time = self.start_time
        self.end_time = clock_data["end_time"]
        self.this_step = 0
        self.num_steps = clock_data["num_steps"]
        self.dt = ((clock_data["end_time"] - clock_data["start_time"]) /
                        clock_data["num_steps"])
        
    def advance(self):
        self.this_step += 1
        self.time = self.start_time + self.dt * self.this_step
    
    def is_running(self):
        return self.this_step <= self.num_steps


class Diagnostic:
    diagnostic_library = {}
    
    @classmethod
    def add_diagnostic_to_library(cls, diagnostic_name, diagnostic_class):
        if diagnostic_name in cls.diagnostic_library:
            raise ValueError("Diagnositc '{0}' already in diagnositc library".format(diagnostic_name))
        cls.diagnostic_library[diagnostic_name] = diagnostic_class

    @classmethod
    def lookup_name(cls, diagnostic_name):
        try:
            return cls.diagnostic_library[diagnostic_name]
        except KeyError:
            raise KeyError("Diagnositc '{0}' not found in diagnositc library".format(diagnostic_name))
            
    def __init__(self, owner: Simulation, input_data: dict):
        self.owner = owner
        self.input_data = input_data

    def inspect_resource(self, resource: dict):
        pass
        
    def diagnose(self):
        raise NotImplementedError
    
    def initialize(self):
        pass
    
    def finalize(self):
        pass


class PointDiagnostic(Diagnostic):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.location = input_data["location"]
        self.field_name = input_data["field"]
        self.output = input_data["output"] # "stdout"
        self.idx = 0
        self.field = None
        
    def diagnose(self):
        self.output_function(self.field[self.idx])

    def inspect_resource(self, resource):
        if self.field_name in resource:
            self.field = resource[self.field_name]
    
    def initialize(self):
        # check that point is within grid
        # set idx to the closest index point
        
        # setup output method
        self.output_function = print


Diagnostic.add_diagnostic_to_library("point", PointDiagnostic)


class Grid:
    def __init__(self, grid_data: dict):
        self.grid_data = grid_data
        self.num_points = grid_data["N"]
        self.r_min = grid_data["r_min"]
        self.r_max = grid_data["r_max"]
        self.r = self.r_min + self.r_max * self.generate_linear()
    
    def generate_field(self, num_components=1):
        return np.zeros((self.num_points, num_components))
    
    def generate_linear(self):
        return np.linspace(0, 1, self.num_points)
        
