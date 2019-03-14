import numpy as np
from scipy import interpolate

def findall(string, sub):
    """
    Find all indices where a substring occurs in a string
    """
    index = 0 - len(sub)
    i = []
    try:
        char_string = np.ndarray.tolist(string)
    except:
        char_string = string
    try:
        while True:
            index = char_string.index(sub, index + len(sub))
            i.append( index )
        return np.array(i)
    except ValueError:
        return np.array(i)
        pass
    return None

class Reactions:
    def __init__(self):
        self.reactants = []
        self.products = []
        self.type = ' '
        self.energy = 0.0
        self.rate_coeff = 0.0
        self.LOG10_k = 0.0
        self.Z = 1.0
        self.A = 1.0
        self.deltaE = 0.0
        self.rx_rate   = None
        self.mom_rate = None
        self.erg_rate = None
        
        return


class Reaction:
    def __init__(self):
#         self.reactants = []
#         self.products = []
#         self.type = ' '
#         self.energy = 0.0
#         self.rate_coeff = 0.0
#         self.LOG10_k = 0.0
#         self.Z = 1.0
#         self.A = 1.0
#         self.deltaE = 0.0
#         self.rx_rate   = None
#         self.mom_rate = None
#         self.erg_rate = None
#         
#         return
        pass

class FluidSpecies:
    def __init__(self, owner, species, ic, immobile=True):
        self.species = species
        self.owner = owner
        self.density = None
        self.velocity = None
        self.energy = None
        self.type = ' '
        self.A = 1
        self.Z = 1
        self.index = 0
        
        self.set_conditions(ic)        
    
    def set_conditions(ic):
        self.density = ic.density * self.owner.grid.generate_ones_centered_field(1)
        if not self.immobile:
            self.velocity = ic.velocity * self.owner.grid.generate_ones_centered_field(1)



class Chemistry:

    def __init__(self, RateFileList):
        self.tiny = np.finfo(1.0).tiny
        self.rate_files = RateFileList
        
        self.parse_files(self.rate_files)
        
    def create_species(self):
        pass
        
    def plasma_chemistry(self,fname_list):
        """
        This function takes a list of reaction-rate files and determines the 
        unique set of species.
        This function returns two important data structures that are used in the chemistry
        bookeeping.
        
        self.chemistry  - is a dictionary that keeps tract of the chemical reactions.  The dictionary keys are 
                          the reactions.  Each dictionary item is a data structure of type Reactions.
                          The data structure Reactions has the following entries
                          
                          Reactions object:
                           
                              reactants  -  a Species object for each of the reactants
                              products   -  a Species object for each of the products
                              type       -  reaction type (excitation, ionization, etc)
                              energy     -  tabulated energies for the reaction-rate
                              rate_coeff -  tabulated reaction-rate
                              LOG10_k    -  an interpolating function for the log10(k) supply and energy and it 
                                            returns the log10(k).  This is the interpolation workhorse.
                              Z          -  a string indicating the charge state changtes (if any) for the reaction
                              A          -  a string indicating the atomic number for the reaction
                              deltaE     -  the energy threshold for the reaction
                              rx_rate    -  This is the interpolated rx_rate
                              mom_rate   -  This is the interpolated mom_rate
                              erg_rate   -  This is the interpolated energy_rate

                    
        self.species    - is a dictionary species objects for each unique species
        
                          Species object:
                              species_name  - a unique string identifier of the species
                              density       - a numpy array for the species density
                              velocity      - a numpy array for the species velocity
                              energy        - a numpy array for the species energy
                              type          - a string identifying the species type
                              A             - the atomic mass of the species
                              Z             - the charge state of the species
                              index         - an integer that identifies the species
        """
        ptype = fname_list.__class__
        if ptype==str:
            chemistry_fnames = [fname_list]
        elif ptype==np.ndarray:
            chemistry_fnames = [np.ndarray.tolist(fname_list)]
        elif ptype ==list:
            chemistry_fnames = fname_list

#  Parse the reaction rates files
#
        fname = chemistry_fnames[0]
        rx = self.readfile(fname)
        species, chemistry = self.CreateChemistry(rx)
        RX = chemistry
        for fname in chemistry_fnames[1:]:
            rx = self.readfile(fname)
            SP, R1 = self.CreateChemistry(rx)
            species, chemistry = self.AddReactions(RX,R1)
        nsp = species.__len__()
        self.species = species
        self.chemistry = chemistry
        return
        
    def get_coefficient(self, reaction, electron_energy):
        """
        This function returns the transport rate coefficient for the given reaction
        and given electron energy.  The shape of the returned rate coefficient matches
        the shape of the passed electron energy array.
        """
        chem = self.chemistry
        x = np.log10(electron_energy)
        k = 10.0**chem[reaction].LOG10_k(x)
        return k
        
    def RHS(self, species):
        """
        This function computes the source and sink terms for the plasma chemistry
        """
        chem = self.chemistry
        reactions = chem.keys()
        electron_energy = species['e'].energy
        dndt = {}
        nu_m = {}
        dedt = {}
        for sp in species.keys():
            dndt[sp] = 0.0
            nu_m[sp] = 0.0
            dedt[sp] = 0.0
        mom_rate = 0.0
        for reaction in reactions:
            k = 1E-6 * self.get_coefficient(reaction, electron_energy)
            threshold_energy = chem[reaction].deltaE
            reacting_species = chem[reaction].reactants
            n1 = reacting_species[0].density
            n2 = reacting_species[1].density
            reaction_rate = k * n1 * n2
            energy_rate = reaction_rate*threshold_energy
            if chem[reaction].type == 'Ground/MomXfer':
                mom_rate = mom_rate + reaction_rate
            for RX in reacting_species:
                dndt[RX.species_name] = dndt[RX.species_name] - reaction_rate
                if  RX.species_name == 'e':
                    dedt['e'] = dedt['e'] - reaction_rate*threshold_energy

            products = chem[reaction].products
            for PR in products:
                dndt[PR.species_name] = dndt[PR.species_name] + reaction_rate
        dedt['e'] = dedt['e']/species['e'].density
        nu_m['e'] = mom_rate/species['e'].density
        rhs = {}
        rhs['density'] = dndt
        rhs['nu_m'] = nu_m
        rhs['energy'] = dedt
        return rhs

    def RHS1(self, species):
        """
        This function computes the source and sink terms for the plasma chemistry
        """
        chem = self.chemistry
        reactions = chem.keys()
        rhs = {}
        
        for sp in species.keys():
            rhs[s]={"density":0,"nu_m":0.0,"energy":0.0}
#            rhs[sp].density = 0.0
#            rhs[sp].nu_m = 0.0
#            rhs[sp].energy= 0.0
        mom_rate = 0.0
        dedt = 0.0
        electron_energy = species['e'].energy
#
#  This loop over reactions computes the RHS for the density equation for all species and 
#  the RHS for the velocity and energy equations for the electrons only.
#
        for reaction in reactions:
            k = 1E-6*self.get_coefficient( reaction, electron_energy ) 
            threshold_energy = chem[reaction].deltaE
#  Reactants
            reacting_species = chem[reaction].reactants
            n1 = reacting_species[0].density
            n2 = reacting_species[1].density
            reaction_rate = k * n1 * n2
            energy_rate = reaction_rate*threshold_energy
            if chem[reaction].type == 'Ground/MomXfer':
                mom_rate = mom_rate + reaction_rate
            for RX in reacting_species:
                rhs[RX.species_name]["density"] = rhs[RX.species_name]["density"] - reaction_rate
                
                if  RX.species_name=='e':
                    dedt = dedt - energy_rate/species['e'].density

#  Products
            products = chem[reaction].products
            for PR in products:
                rhs[PR.species_name]["density"] = rhs[PR.species_name]["density"] + reaction_rate
        rhs['e']["nu_m"] = mom_rate/species['e'].density
        rhs['e']["energy"] = dedt/species['e'].density
        return rhs

    def readfile(self,fname):
        f=open(fname)
        a = f.readlines()
        reactions = []
        types = []
        deltaE = []
        A = []
        Z = []
        rate_coeff = []
        energy = []
        RX = 
        nlines = len(a)
        
        for iline in range(nlines):
            line = a[iline]
            if line[0]=='#' or line[0:5]=='-----':
                pass
            else:
                Ncolon = line.find(':')
                if line[0:Ncolon]=='Reaction':
                    reactions.append( line[Ncolon+1:-1].strip() )
                if line[0:Ncolon]=='Type':
                    types.append(line[Ncolon+1:-1].strip())
                if line[0:Ncolon]=='Delta(eV)':
                    deltaE.append(float(line[Ncolon+1:-1]))
                if line[0:Ncolon]=='Mass(AMU)':
                    A.append(line[Ncolon+1:-1].strip())
                if line[0:Ncolon]=='Charge':
                    Z.append(line[Ncolon+1:-1].strip())
                if line.count('<E>(eV)')>0:
                    erg = []
                    data = []
                    iline = iline + 1
                    while True:
                        if a[iline][0:5]=='-----' or iline>=nlines:
                            break
                        erg.append( float( a[iline].split()[0] ) )
                        data.append( float( a[iline].split()[1] ) )
                        iline = iline + 1
                    rate_coeff.append(data)
                    energy.append(erg)
            
            
        RX.reactions = reactions
        RX.types = types
        RX.deltaE = np.array(deltaE)
        RX.A = A
        RX.Z = Z
        RX.rate_coeff = np.array(rate_coeff)
        RX.energy = np.array(energy)
        RX.reactants = self.getReactants(reactions)
        RX.products = self.getProducts(reactions)

        
        return RX
        
    def getReactants(self,reactions):
        reactants = []
        for reaction in reactions:
            i1 = str.find(reaction,'->')
            LHS = reaction[0:i1]
            i1 = findall(LHS,'+')
            i0 = 0
            temp = []
            for i in i1:
                item = LHS[i0:i]
                temp.append(item)
                i0 = i + 1
            last_item = LHS[i0:]
            temp.append(last_item)
            reactants.append(temp)
        return reactants

    def getProducts(self,reactions):
        products = []
        for reaction in reactions:
            i1 = str.find(reaction,'->')
            RHS = reaction[i1+2:]
            i1 = findall(RHS,'+')
            i0 = 0
            temp = []
            for i in i1:
                item = RHS[i0:i]
                temp.append(item)
                i0 = i + 1
            last_item = RHS[i0:]
            temp.append(last_item)
            products.append(temp)
        return products

    def Species(self,species, Z, A, type):
        s = {}
        keys = species
        ntotal = len( keys )
        for n in range(ntotal):
            key = keys[n]
            attributes = Species()
            attributes.type = type[n]
            attributes.A  = A[n]
            attributes.Z  = Z[n]
            attributes.index = n
            attributes.species_name = key
            s[key] = attributes
        return s

    def CreateChemistry(self,name):
        ntotal = len(name.reactions)
        chem = {}
        for n in range(ntotal):
            RX = Reactions()
            RX.reactants = name.reactants[n]
            RX.products = name.products[n]
            RX.type = name.types[n]
            RX.deltaE = name.deltaE[n]
            RX.A = name.A[n]
            RX.Z = name.Z[n]
            RX.energy = name.energy[n]
            RX.rate_coeff = name.rate_coeff[n]
            x = np.log10(RX.energy)
            y = np.log10(RX.rate_coeff+self.tiny)
            LOG10_k      = interpolate.interp1d(x,y,bounds_error=False,fill_value=[y[0]])
            RX.LOG10_k  = LOG10_k
            key = name.reactions[n]
            chem[key] = RX
        self.reactions = chem.keys()
        species = self.__get_species_info__(chem)
        self.__get_rx_info__(species, chem)
        return species, chem
                
    def AddReactions(self,chem1,chem2):        
        chem = chem1
        for key in chem2.keys():
            if chem1.has_key(key):
                print ('duplicate reaction "', key,'" found')
            else:
                chem[key] = chem2[key]
        self.reactions = chem.keys()        
        species = self.__get_species_info__(chem)
        self.__get_rx_info__(species,chem)
        return species, chem
 
    def __get_rx_info__(self,species,chem):
        reactions = chem.keys()
        SPECIES = np.array(list(species.keys()))
        indices = np.arange( SPECIES.size )
        n = 0
        for reaction in reactions:
            reactants = np.array(chem[reaction].reactants)
            temp=[]
            for r in reactants:
                temp.append( species[r] )
            chem[reaction].reactants = np.array(temp)

            products  = np.array(chem[reaction].products)
            temp=[]
            for p in products:
                temp.append( species[p] )
            chem[reaction].products = np.array(temp)

        return
        
    def __get_species_info__(self,chem):
        species = []
        type = []
        A = []
        Z = []
        for key in chem.keys():
            for item in chem[key].products:
                count = species.count(item)
                if count==0:
                    i = np.array(chem[key].products)==item
                    zz = self.__get_A_or_Z__(chem[key].Z,product=True)
                    Z.append(np.array(zz)[i][0])
                    aa = self.__get_A_or_Z__(chem[key].A,product=True)
                    A.append(np.array(aa)[i][0])
                    species.append(item)
                    type.append(chem[key].type)

            for item in chem[key].reactants:
                count = species.count(item)
                if count==0:
                    Z.append(chem[key].Z[chem[key].reactants==item])
                    species.append(item)
                    type.append(chem[key].type)

        type[species=='e'] = 'Electron'
        species = np.array(species)
        Z = np.array(Z)
        A = np.array(A)
        type = np.array(type)
        species = self.Species(species, Z, A, type)
        return species
        
    def __get_A_or_Z__(self,A,product=True):
        a = []
        AA = A
        i1 = str.find(AA,'->')
        if product:
            item = AA[i1+2:]
        else:
            item = AA[0:i1]
        ii = findall(item,':')
        i0 = 1
        for i in ii:
            f = float(item[i0:i])
            a.append(f)
            i0 = i + 1
        last_item = float(item[i+1:-1])
        a.append(last_item) 
        return a        
            
