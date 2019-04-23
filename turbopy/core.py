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
        if "Tools" in self.input_data:
            for t in self.input_data["Tools"]:
                tool_class = ComputeTool.lookup(t["type"])
                # todo: somehow make tool names unique, or prevent more than one each
                self.compute_tools.append(tool_class(owner=self, input_data=t))

    def read_modules_from_input(self):
        for module_data in self.input_data["Modules"]:
            module_class = Module.lookup(module_data["name"])
            self.modules.append(module_class(owner=self, input_data=module_data))
        self.sort_modules()
    
    def read_diagnostics_from_input(self):
        if "Diagnostics" in self.input_data:
            for d in self.input_data["Diagnostics"]:
                diagnostic_class = Diagnostic.lookup(d["type"])
                self.diagnostics.append(diagnostic_class(owner=self, input_data=d))
    
    def sort_modules(self):
        pass
    
    def find_tool_by_name(self, tool_name):
        tools = [t for t in self.compute_tools if t.name == tool_name]
        if len(tools) == 1:
            return tools[0]
        return None
            

class DynamicFactory:
    """
    This base class provides a dynamic factory pattern functionality to classes 
    that derive from this.
    """
    _registry = {}
    _factory_type_name = "Class"
    
    @classmethod
    def register(cls, name_to_register, class_to_register):
        if name_to_register in cls._registry:
            raise ValueError("{0} '{1}' already registered".format(cls._factory_type_name, name_to_register))
        cls._registry[name_to_register] = class_to_register

    @classmethod
    def lookup(cls, name):
        try:
            return cls._registry[name]
        except KeyError:
            raise KeyError("{0} '{1}' not found in registry".format(cls._factory_type_name, name))



class Module(DynamicFactory):
    """
    This is the base class for all physics modules
    Based on Module class in TurboWAVE

    Because python mutable/immutable is different than C++ pointers, the implementation 
    here is different. Here, a "resource" is a dictionary, and can have more than one 
    thing being shared. Note that the value stored in the dictionary needs to be mutable. 
    Make sure not to reinitialize, because other modules will be holding a reference to it.
    """
    _factory_type_name = "Module"
    
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
        

class ComputeTool(DynamicFactory):
    """
    This is the base class for compute tools. These are the compute-heavy functions,
    which have implementations of numerical methods which can be shared between modules.
    """
    _factory_type_name = "Compute Tool"
    
    def __init__(self, owner: Simulation, input_data: dict):
        self.owner = owner
        self.input_data = input_data
        self.name = input_data["type"]
    
    def initialize(self):
        pass


class PoissonSolver1DRadial(ComputeTool):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.field = None
        
    def initialize(self):
        self.field = self.owner.grid.generate_field(1)
    
    def solve(self, sources):
        r = self.owner.grid.r
        dr = np.mean(self.owner.grid.cell_widths)
        I1 = np.cumsum(r * sources * dr)
        integrand = I1 * dr / r
        i0 = 2 * integrand[1] - integrand[2]   # linearly extrapolate to r = 0
        integrand[0] = i0
        integrand = integrand - i0     # add const of integr so derivative = 0 at r = 0
        I2 = np.cumsum(integrand)
        return I2 - I2[-1]


ComputeTool.register("PoissonSolver1DRadial", PoissonSolver1DRadial)


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
        return self.this_step < self.num_steps


class Diagnostic(DynamicFactory):
    _factory_type_name = "Diagnostic"
                
    def __init__(self, owner: Simulation, input_data: dict):
        self.owner = owner
        self.input_data = input_data

    def inspect_resource(self, resource: dict):
        """
        If your subclass needs the data described by the key, now's their chance to 
        save a pointer to the data
        """
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


class FieldDiagnostic(Diagnostic):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        
        self.component = input_data["component"]
        self.field_name = input_data["field"]
        self.output = input_data["output"] # "stdout"
        self.field = None
        
        self.file = None
        
    def diagnose(self):
        self.output_function(self.field[:,self.component])

    def inspect_resource(self, resource):
        if self.field_name in resource:
            self.field = resource[self.field_name]
    
    def print_diag(self, data):
        print(self.field_name, data)
        
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
        self.outputbuffer[i,:] = data[:]
    
    def finalize(self):
        self.diagnose()
        if self.input_data["output"] == "csv":
            self.file = open(self.input_data["filename"], 'wb')
            np.savetxt(self.file, self.outputbuffer, delimiter=",")
            self.file.close()


class GridDiagnostic(Diagnostic):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.filename = "grid.csv"
        if "filename" in input_data:
            self.filename = input_data["filename"]
            
    def diagnose(self):
        pass

    def initialize(self):
        self.file = open(self.filename, 'wb')
        np.savetxt(self.file, self.owner.grid.r, delimiter=",")
        self.file.close()
    
    def finalize(self):
        pass                

Diagnostic.register("point", PointDiagnostic)
Diagnostic.register("field", FieldDiagnostic)
Diagnostic.register("grid", GridDiagnostic)


class Grid:
    def __init__(self, grid_data: dict):
        self.grid_data = grid_data
        self.num_points = grid_data["N"]
        self.r_min = grid_data["r_min"]
        self.r_max = grid_data["r_max"]
        self.r = self.r_min + self.r_max * self.generate_linear()
        self.cell_edges = self.r
        self.cell_centers = (self.r[1:] + self.r[:-1])/2
        self.cell_widths = (self.r[1:] - self.r[:-1])
    
    def generate_field(self, num_components=1):
        return np.zeros((self.num_points, num_components))
    
    def generate_linear(self):
        return np.linspace(0, 1, self.num_points)
        
