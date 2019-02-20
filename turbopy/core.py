# Computational Physics Simulation Framework
#
# Based on the structure of turboWAVE
#

class Simulation:
    """
    This "owns" all the physics modules, and coordinates them
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
        print("Simulation complete")
    
    def fundamental_cycle(self):
        self.diagnose()
        
        for m in self.modules:
            m.reset()
        
        for m in self.modules:
            m.update()
        
        self.clock.advance()
        
    def diagnose(self):
        for d in self.diagnostics:
            if d.write_this_step(self.clock):
                d.diagnose()
    
    def prepare_simulation(self):
        print("Reading Grid...")
        self.read_grid_from_input()
        print("Reading Tools...")
        self.read_tools_from_input()
        print("Reading Modules...")
        self.read_modules_from_input()
        self.sort_modules()
        print("Initializing Tools...")
        for t in self.compute_tools:
            t.initialize()
        print("Initializing Modules...")
        for m in self.modules:
            m.initialize()


class Module:
    """
    This is the base class for all physics modules
    Based on Module class in TurboWAVE

    Because python mutable/immutable is different than C++ pointers, the implementation here is different.
    Here, a "resource" is a dictionary, and can have more than one thing being shared
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
    pass


class SimulationClock:
    pass
