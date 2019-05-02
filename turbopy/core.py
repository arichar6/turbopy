# Computational Physics Simulation Framework
#
# Based on the structure of turboWAVE
#
import numpy as np
import scipy.interpolate as interpolate

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


class FiniteDifference(ComputeTool):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.method = input_data["method"]
        self.dr_centered = self.owner.grid.r[2:] - self.owner.grid.r[:-2]
    
    def setup_ddx(self):
        assert (self.method in ["centered", "upwind_left"])
        if self.method == "centered":
            return self.centered_difference
        if self.method == "upwind_left":
            return self.upwind_left
    
    def centered_difference(self, y):
        d = self.owner.grid.generate_field()
        d[1:-1] = (y[2:] - y[:-2]) / self.dr_centered
        return d
    
    def upwind_left(self, y):
        d = self.owner.grid.generate_field()
        d[1:] = (y[1:] - y[:-1]) / self.owner.grid.cell_widths
        return d
        

class BorisPush(ComputeTool):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.c2 = (2.9979e8)**2

    def push(self, position, momentum, charge, mass, E, B):
        dt = self.owner.clock.dt

        vminus = momentum + dt * E * charge / 2
        m1 = np.sqrt(mass**2 + np.dot(momentum, momentum)/self.c2)

        t = dt * B * charge / m1 / 2
        s = 2 * t / (1 + np.dot(t, t))
        
        vprime = vminus + np.cross(vminus, t)
        vplus = vminus + np.cross(vprime, s)
        momentum[:] = vplus + dt * E * charge / 2
        m2 = np.sqrt(mass**2 + np.dot(momentum, momentum)/self.c2)
        position[:] = position + dt * momentum / m2
        
class Interpolators(ComputeTool):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
    def interpolate1D(self,x,y,kind='linear'):
        f = interpolate.interp1d(x,y,kind)
        return f

ComputeTool.register("BorisPush", BorisPush)
ComputeTool.register("PoissonSolver1DRadial", PoissonSolver1DRadial)
ComputeTool.register("FiniteDifference", FiniteDifference)
ComputeTool.register("Interpolators",Interpolators)

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


class CSVDiagnosticOutput:
    def __init__(self, filename, diagnostic_size):
        self.filename = filename
        self.buffer = np.zeros(diagnostic_size)
        self.file = None
        self.buffer_index = 0
        
    def append(self, data):
        self.buffer[self.buffer_index, :] = data
        self.buffer_index += 1
    
    def finalize(self):
        self.file = open(self.filename, 'wb')
        np.savetxt(self.file, self.buffer, delimiter=",")
        self.file.close()


class PointDiagnostic(Diagnostic):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.location = input_data["location"]
        self.field_name = input_data["field"]
        self.output = input_data["output"] # "stdout"
        self.get_value = None
        self.field = None
                
    def diagnose(self):
        self.output_function(self.get_value(self.field))

    def inspect_resource(self, resource):
        if self.field_name in resource:
            self.field = resource[self.field_name]

    def print_diagnose(self, data):
        print(data)
        
    def initialize(self):
        # set up function to interpolate the field value
        self.get_value = self.owner.grid.create_interpolator(self.location)
        
        # setup output method
        functions = {"stdout": self.print_diagnose,
                     "csv": self.csv_diagnose,
                     }
        self.output_function = functions[self.input_data["output"]]

        if self.input_data["output"] == "csv":
            diagnostic_size = (self.owner.clock.num_steps + 1, 1)
            self.csv = CSVDiagnosticOutput(self.input_data["filename"], diagnostic_size)

    def csv_diagnose(self, data):
        self.csv.append(data)

    def finalize(self):
        self.diagnose()
        if self.input_data["output"] == "csv":
            self.csv.finalize()
            
class FieldDiagnostic(Diagnostic):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        
        self.component = input_data["component"]
        self.field_name = input_data["field"]
        self.output = input_data["output"] # "stdout"
        self.field = None
        
    def diagnose(self):
        if len(self.field.shape) > 1:
            self.output_function(self.field[:,self.component])
        else:
            self.output_function(self.field)

    def inspect_resource(self, resource):
        if self.field_name in resource:
            self.field = resource[self.field_name]
    
    def print_diagnose(self, data):
        print(self.field_name, data)
        
    def initialize(self):
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
        self.dr = self.cell_widths[0] # only good for uniform grids!
    
    def generate_field(self, num_components=1):
        return np.squeeze(np.zeros((self.num_points, num_components)))
    
    def generate_linear(self):
        return np.linspace(0, 1, self.num_points)
    
    def create_interpolator(self, r0):
        # Return a function which linearly interpolates any field on this grid, to the point x
        assert (r0 >= self.r_min), "Requested point is not in the grid"
        assert (r0 <= self.r_max), "Requested point is not in the grid"
        i, = np.where( (r0 - self.dr < self.r) & (self.r < r0 + self.dr))
        assert (len(i) in [1, 2]), "Error finding requested point in the grid"
        if len(i) == 1:
            return lambda y: y[i]
        if len(i) == 2:
            # linearly interpolate
            def interpval(yvec):
                rvals = self.r[i]
                y = yvec[i]
                return y[0] + (r0 - rvals[0]) * (y[1]-y[0])/(rvals[1]-rvals[0])
            return interpval
            


