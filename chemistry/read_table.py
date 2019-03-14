import numpy as np
from scipy import interpolate


class Species:
    def __init__(self, mass, charge, name):
        self.name = name
        self.mass = mass
        self.charge = charge
    
    def __repr__(self):
        return '(' + ', '.join(["Species: " + self.name,
                          "Mass: " + str(self.mass),
                          "Charge: " + str(self.charge)]) + ')'
    
    def __eq__(self, other):
        if self.name == other.name:
            return True
        return False

    def __hash__(self):
        return hash(self.name)


class Reaction:
    def __init__(self, header, data, units='MKS'):
        self.ratedata = self.parse_table(data)
        self.set_interpolating_function()
        self.reactants = []
        self.products = []
        self.type = header['Type']
        self.delta_e = float(header['Delta(eV)'])
        self.identifier = header['Reaction']
        self.interpret_reaction(header)
        if units == 'MKS':
            self.AMU = 1.66054E-027
            self.echarge = 1.602E-19
        else:
            self.AMU = 1
            self.echarge = 1

    
    def __repr__(self):
        return self.type + " Reaction: " + self.identifier

    def parse_table(self, data):
        temp = np.loadtxt(data, skiprows=1)
        energy = temp[:,0]
        rate =   temp[:,1]
        return energy,rate
        
    def set_interpolating_function(self):
        energy,rate = self.ratedata
        x = np.log10(energy)
        y = np.log10(rate+np.finfo(1.0).tiny)
        self.LOG10_k = interpolate.interp1d(x,y,bounds_error=False,fill_value=[y[0]])
        return
        
    def get_rate(self,energy):
        x = np.log10(energy)
        y = self.LOG10_k()
        k = 10**y
        return k
    
    def interpret_reaction(self, header):
        [mass_r, mass_p] = [m.strip('()').split(':') for m in header['Mass(AMU)'].split('->')]
        [charge_r, charge_p] = [q.strip('()').split(':') for q in header['Charge'].split('->')]
        [name_r, name_p] = [n.split('+') for n in header['Reaction'].split('->')]
        
        for m, q, name in zip(mass_r, charge_r, name_r):
            mass  = self.AMU*float(m)
            charge = self.echarge*float(q)
            self.reactants.append(Species(float(m), float(q), name))

        for m, q, name in zip(mass_p, charge_p, name_p):
            self.products.append(Species(float(m), float(q), name))

    def __eq__(self, other):
        if self.identifier == other.identifier:
            return True
        return False

    def __hash__(self):
        return hash(self.identifier)
        
class Chemistry:
    def __init__(self):
        self.RateFileList = RateFileList
        self.RX_list = []
#        self.chem = self.CreateChemistry()
#        if qval!=None:
#            self.Renormalize_charge(chem,qval)
#        if mval!=None:
#            self.Renormalize_mass(chem,mval)
        return
        
    def CreateChemistry(self,RateFileList):
        chem = []
        for ratefile in self.RateFileList:
            reactions = parse_rate_file(ratefile)
            for reaction in reactions:
                if not reaction in self.RX_list:
                    chem.append(reaction)
                    self.RX_list.append(reaction)
                else:
                    print('Warning:  Duplicate reaction',r,' found')
        self.chem = chem
        return chem
        
    def UniqueSpeciesList(self, chem):
        species=[]
        for reaction in chem:
            for sp in reaction.reactants:
                if not sp in species:
                    species.append(sp)
            for sp in reaction.products:
                if not sp in species:
                    species.append(sp)
        

def read_section(fp):
    separator, section = None, []
    for line in fp:
        line = line.rstrip()
        if line.startswith("-----"):
            if separator: yield section
            separator = line
            section = []
        else:
            section.append(line)
    if section: yield section

def read_reaction(sections):
    header, data = None, None
    j = 0
    for s in sections:
        if j % 2:
            data = s
            yield(header, data)
        else:
            header = s
        j = j + 1

def parse_header_to_dict(header):
    header_data = {}
    for line in header:
        if not line.startswith('#'):
            line_data = [item.strip() for item in line.split(':', 1)]
            header_data[line_data[0]] = line_data[1]
    return header_data

def parse_rate_file(fname):
    reactions = []
    with open(fname,'r') as f:
        for h, d in read_reaction(read_section(f)):
            h = parse_header_to_dict(h)
            r = Reaction(h, d)
            reactions.append(r)
    return reactions





