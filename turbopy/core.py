# Computational Physics Simulation Framework
#
# Based on the structure of turboWAVE
#
import numpy as np
import scipy.interpolate as interpolate
from scipy import sparse
import qtoml as toml
from pathlib import Path

class Simulation:
    """
    This "owns" all the physics modules and compute tools, and coordinates them.
    The main simulation loop is driven by an instance of this class.
    
    Based on the Simulation class in TurboWAVE
    """
    def __init__(self, input_data: dict):
        # Check if the input is a filename of parameters to parse
        if isinstance(input_data, str):
            with open(input_data) as f:
                input_data = toml.load(f)
        
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
            for tool_name, params in self.input_data["Tools"].items():
                tool_class = ComputeTool.lookup(tool_name)
                params["type"] = tool_name
                # todo: somehow make tool names unique, or prevent more than one each
                self.compute_tools.append(tool_class(owner=self, input_data=params))

    def read_modules_from_input(self):
        for module_name, module_data in self.input_data["Modules"].items():
            module_class = Module.lookup(module_name)
            module_data["name"] = module_name
            self.modules.append(module_class(owner=self, input_data=module_data))
        self.sort_modules()
    
    def read_diagnostics_from_input(self):
        if "Diagnostics" in self.input_data:
            # This dictionary has two types of keys:
            #    keys that are valid diagnostic types
            #    other keys, which should be passed along as "default" parameters
            diags = {k:v for k,v in self.input_data["Diagnostics"].items() if Diagnostic.is_valid_name(k)}
            params = {k:v for k,v in self.input_data["Diagnostics"].items() if not Diagnostic.is_valid_name(k)}
            
            # todo: implement a system for making default file names
            if "directory" in params:
                d = Path(params["directory"])
                d.mkdir(parents=True, exist_ok=True)
            
            for diag_type, d in diags.items():
                diagnostic_class = Diagnostic.lookup(diag_type)
                if not type(d) is list:
                    d = [d]
                for di in d:
                    # Values in di supersede values in params because of the order
                    di = {**params, **di, "type": diag_type}
                    if "directory" in di and "filename" in di:
                        di["filename"] = str(Path(di["directory"]) / Path(di["filename"]))
                    self.diagnostics.append(diagnostic_class(owner=self, input_data=di))
    
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
    
    @classmethod
    def is_valid_name(cls, name):
        return name in cls._registry



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
    _registry = {}
    
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
    _registry = {}
    
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
        self.dr = self.owner.grid.dr
    
    def setup_ddx(self):
        assert (self.input_data["method"] in ["centered", "upwind_left"])
        if self.input_data["method"] == "centered":
            return self.centered_difference
        if self.input_data["method"] == "upwind_left":
            return self.upwind_left
    
    def centered_difference(self, y):
        d = self.owner.grid.generate_field()
        d[1:-1] = (y[2:] - y[:-2]) / self.dr_centered
        return d
    
    def upwind_left(self, y):
        d = self.owner.grid.generate_field()
        d[1:] = (y[1:] - y[:-1]) / self.owner.grid.cell_widths
        return d

    def radial_curl(self):
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
        col_above[1] = 2.0 / self.dr     # for col_above, the first element is dropped
        # At r=Rw, use rB~const?
        col_diag[-1] = 1.0 / self.dr     # for col_below, the last element is dropped
        col_below[-2] = 2.0 * col_below[-1]
        # set main columns for finite difference derivative
        D = sparse.dia_matrix( ([col_below, col_diag, col_above], [-1, 0, 1]), shape=(N, N) )
        return D
    
    def del2_radial(self):
        # FD matrix for (1/r)(d/dr)(r (df/dr))
        N = self.owner.grid.num_points
        g1 = 1/(2.0 * self.dr)
        col_below = -g1 * np.ones(N)
        col_above = g1 * np.ones(N)
        
        col_above[1:] = self.owner.grid.r[:-1] * col_above[1:]
        col_below[:-1] = self.owner.grid.r[1:] * col_below[:-1]
        
        # BC at r=0
        col_above[1] = 0
        
        D1 = sparse.dia_matrix(([col_below, col_above], [-1, 1]), shape=(N, N))
        
        g2 = 1/(self.dr**2)
        col_below = g2 * np.ones(N)
        col_diag = g2 * np.ones(N)
        col_above = g2 * np.ones(N)
        
        # BC at r=0, first row of D
        col_above[1] = 2 * col_above[1]
        D2 = sparse.dia_matrix(([col_below, -2*col_diag, col_above], [-1, 0, 1]), shape=(N, N))
        
        # Need to set boundary conditions!
        D = D1 + D2
        return D
    
    def ddr(self):
        # FD matrix for (d/dr) f
        N = self.owner.grid.num_points
        g1 = 1/(2.0 * self.dr)
        col_below = -g1 * np.ones(N)
        col_above = g1 * np.ones(N)
        # BC at r=0
        col_above[1] = 0
        D1 = sparse.dia_matrix(([col_below, col_above], [-1, 1]), shape=(N, N))        
        return D1

    def BC_left_extrap(self):
        N = self.owner.grid.num_points
        col_diag = np.ones(N)
        col_above = np.zeros(N)
        col_above2 = np.zeros(N)
        
        # for col_above, the first element is dropped
        col_diag[0] = 0
        col_above[1] = 2
        col_above2[2] = -1

        BC = sparse.dia_matrix(([col_diag, col_above, col_above2], [0,1,2]), shape=(N, N))
        return BC

    def BC_left_avg(self):
        N = self.owner.grid.num_points
        col_diag = np.ones(N)
        col_above = np.zeros(N)
        col_above2 = np.zeros(N)
        
        # for col_above, the first element is dropped
        col_diag[0] = 0
        col_above[1] = 1.5
        col_above2[2] = -0.5

        BC = sparse.dia_matrix(([col_diag, col_above, col_above2], [0,1,2]), shape=(N, N))
        return BC        

    def BC_left_quad(self):
        N = self.owner.grid.num_points
        r = self.owner.grid.r
        col_diag = np.ones(N)
        col_above = np.zeros(N)
        col_above2 = np.zeros(N)
        
        R2 = (r[1]**2 + r[2]**2)/(r[2]**2 - r[1]**2)/2
        # for col_above, the first element is dropped
        col_diag[0] = 0
        col_above[1] = 0.5 + R2
        col_above2[2] = 0.5 - R2

        BC = sparse.dia_matrix(([col_diag, col_above, col_above2],
                                [0, 1, 2]), shape=(N, N))
        return BC
    
    def BC_left_flat(self):
        N = self.owner.grid.num_points
        col_diag = np.ones(N)
        col_above = np.zeros(N)
        col_above2 = np.zeros(N)
        # for col_above, the first element is dropped
        col_diag[0] = 0
        col_above[1] = 1

        BC = sparse.dia_matrix(([col_diag, col_above], [0,1]), shape=(N, N))
        return BC        
    
    def BC_right_extrap(self):
        N = self.owner.grid.num_points
        col_diag = np.ones(N)
        col_below = np.zeros(N)
        col_below2 = np.zeros(N)
        
        # for col_below, the last element is dropped
        col_diag[-1] = 0
        col_below[-2] = 2
        col_below2[-3] = -1

        BC_right = sparse.dia_matrix(([col_below2, col_below, col_diag], [-2, -1, 0]), shape=(N, N))
        return BC_right


class BorisPush(ComputeTool):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.c2 = 2.9979e8 ** 2

    def push(self, position, momentum, charge, mass, E, B):
        dt = self.owner.clock.dt

        vminus = momentum + dt * E * charge / 2
        m1 = np.sqrt(mass**2 + np.sum(momentum*momentum, axis=-1)/self.c2)

        t = dt * B * charge / m1[:, np.newaxis] / 2
        s = 2 * t / (1 + np.sum(t*t, axis=-1)[:, np.newaxis])
        
        vprime = vminus + np.cross(vminus, t)
        vplus = vminus + np.cross(vprime, s)
        momentum[:] = vplus + dt * E * charge / 2
        m2 = np.sqrt(mass**2 + np.sum(momentum*momentum, axis=-1)/self.c2)
        position[:] = position + dt * momentum / m2[:, np.newaxis]


class Interpolators(ComputeTool):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)

    def interpolate1D(self, x, y, kind='linear'):
        f = interpolate.interp1d(x, y, kind)
        return f


ComputeTool.register("BorisPush", BorisPush)
ComputeTool.register("PoissonSolver1DRadial", PoissonSolver1DRadial)
ComputeTool.register("FiniteDifference", FiniteDifference)
ComputeTool.register("Interpolators", Interpolators)


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
    _registry = {}
                
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
        self.buffer_index = 0
        
    def append(self, data):
        self.buffer[self.buffer_index, :] = data
        self.buffer_index += 1
    
    def finalize(self):
        with open(self.filename, 'wb') as f:
            np.savetxt(f, self.buffer, delimiter=",")


class PointDiagnostic(Diagnostic):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.location = input_data["location"]
        self.field_name = input_data["field"]
        self.output = input_data["output_type"] # "stdout"
        self.get_value = None
        self.field = None
        self.output_function = None
        self.csv = None
                
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
        self.output_function = functions[self.input_data["output_type"]]

        if self.input_data["output_type"] == "csv":
            diagnostic_size = (self.owner.clock.num_steps + 1, 1)
            self.csv = CSVDiagnosticOutput(self.input_data["filename"], diagnostic_size)

    def csv_diagnose(self, data):
        self.csv.append(data)

    def finalize(self):
        self.diagnose()
        if self.input_data["output_type"] == "csv":
            self.csv.finalize()


class FieldDiagnostic(Diagnostic):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        
        self.component = input_data["component"]
        self.field_name = input_data["field"]
        self.output = input_data["output_type"] # "stdout"
        self.field = None

        self.dump_interval = None        
        self.diagnose = self.do_diagnostic
        self.diagnostic_size = None


    def check_step(self):
        if (self.owner.clock.time >= self.last_dump + self.dump_interval):
            self.do_diagnostic()
            self.last_dump = self.owner.clock.time
    
    def do_diagnostic(self):
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
        self.diagnostic_size = (self.owner.clock.num_steps+1,
                                self.owner.grid.num_points)
        if "dump_interval" in self.input_data:
            self.dump_interval = self.input_data["dump_interval"]
            self.diagnose = self.check_step
            self.last_dump = 0
            self.diagnostic_size = (int(np.ceil(self.owner.clock.end_time/self.dump_interval)+1),
                                    self.owner.grid.num_points)       
    
        # setup output method
        functions = {"stdout": self.print_diagnose,
                     "csv": self.csv_diagnose,
                     }
        self.output_function = functions[self.input_data["output_type"]]
        if self.input_data["output_type"] == "csv":
            self.csv = CSVDiagnosticOutput(self.input_data["filename"], self.diagnostic_size)
    
    def csv_diagnose(self, data):
        self.csv.append(data)
    
    def finalize(self):
        self.do_diagnostic()
        if self.input_data["output_type"] == "csv":
            self.csv.finalize()


class GridDiagnostic(Diagnostic):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.filename = input_data["filename"]
            
    def diagnose(self):
        pass

    def initialize(self):
        with open(self.filename, 'wb') as f:
            np.savetxt(f, self.owner.grid.r, delimiter=",")

    def finalize(self):
        pass


class ClockDiagnostic(Diagnostic):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.filename = input_data["filename"]
        self.csv = None

    def diagnose(self):
        self.csv.append(self.owner.clock.time)

    def initialize(self):
        diagnostic_size = (self.owner.clock.num_steps + 1, 1)
        self.csv = CSVDiagnosticOutput(self.input_data["filename"], diagnostic_size)

    def finalize(self):
        self.diagnose()
        self.csv.finalize()


Diagnostic.register("point", PointDiagnostic)
Diagnostic.register("field", FieldDiagnostic)
Diagnostic.register("grid", GridDiagnostic)
Diagnostic.register("clock", ClockDiagnostic)


class Grid:
    def __init__(self, grid_data: dict):
        self.grid_data = grid_data
        self.num_points = grid_data["N"]
        self.r_min = grid_data["r_min"]
        self.r_max = grid_data["r_max"]
        self.r = self.r_min + (self.r_max - self.r_min) * self.generate_linear()
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
            


