from pprint import pprint
import numpy as np


class Species:
    def __init__(self, mass, charge, name):
        self.name = name
        self.mass = mass
        self.charge = charge
    
    def __repr__(self):
        return '(' + ', '.join(["Species: " + self.name,
                          "Mass: " + str(self.mass),
                          "Charge: " + str(self.charge)]) + ')'


class Reaction:
    def __init__(self, header, data):
        self.data = self.parse_table(data)
        self.reactants = []
        self.products = []
        self.type = header['Type']
        self.delta_e = header['Delta(eV)']
        self.identifier = header['Reaction']
        self.interpret_reaction(header)
    
    def __repr__(self):
        return self.type + " Reaction: " + self.identifier

    def parse_table(self, data):
        return np.loadtxt(data, skiprows=1)
    
    def interpret_reaction(self, header):
        [mass_r, mass_p] = [m.strip('()').split(':') for m in header['Mass(AMU)'].split('->')]
        [charge_r, charge_p] = [q.strip('()').split(':') for q in header['Charge'].split('->')]
        [name_r, name_p] = [n.split('+') for n in header['Reaction'].split('->')]
        
        for m, q, name in zip(mass_r, charge_r, name_r):
            self.reactants.append(Species(float(m), float(q), name))

        for m, q, name in zip(mass_p, charge_p, name_p):
            self.products.append(Species(float(m), float(q), name))


def read_section(fp):
    separator, section = None, []
    for line in fp:
        line = line.rstrip()
        if line.startswith("-----"):
            if separator: yield section
            separator, section = line, []
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

# Open and parse a rates table as an example of how this code works
fname = 'chemistry/N2_Rates_TT.txt'

with open(fname) as f:
    for h, d in read_reaction(read_section(f)):
        h = parse_header_to_dict(h)
        r = Reaction(h, d)
        print(r)
        print(r.reactants, '->', r.products)
