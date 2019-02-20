# Computational Physics Simulation Framework
#
# Based on the structure of turboWAVE
#

class Simulation:
    """
    This "owns" all the physics modules and compute tools, and coordinates them.
    The main simulation loop is driven by an instance of this class.
    
    Based on the Simulation class in TurboWAVE
    """
    def __init__(self):
        self.modules = []
        self.compute_tools = []
        self.diagnostics = []

        self.grid = None
        self.clock = None
        self.units = None
    
    def run(self):
        print("Simulation is initializing")
        self.prepare_simulation()
        print("Initialization complete")
        
        print("Simulation is starting")
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
        raise NotImplementedError
    
    def read_tools_from_input(self):
        raise NotImplementedError
    
    def read_modules_from_input(self):
        raise NotImplementedError
    
    def read_diagnostics_from_input(self):
        raise NotImplementedError
            

class Module:
    """
    This is the base class for all physics modules
    Based on Module class in TurboWAVE

    Because python mutable/immutable is different than C++ pointers, the implementation 
    here is different. Here, a "resource" is a dictionary, and can have more than one 
    thing being shared. Note that the value stored in the dictionary needs to be mutable. 
    Make sure not to reinitialize, because other modules will be holding a reference to it.
    """
    def __init__(self, owner: Simulation):
        self.owner = owner

    def publish_resource(self, resource: dict):
        for module in self.owner.modules:
            module.inspect_resource(resource)

    def inspect_resource(self, resource: dict):
        """If your subclass needs the data described by the key, now's their chance to save a pointer to the data"""
        raise NotImplementedError

    def exchange_resources(self):
        """This is the function where you call publish_resource, to tell other modules about data you want to share"""
        pass

    def update(self):
        raise NotImplementedError
        

class ComputeTool:
    raise NotImplementedError


class SimulationClock:
    raise NotImplementedError


class Diagnostic:
    raise NotImplementedError

