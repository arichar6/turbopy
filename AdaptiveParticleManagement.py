import numpy as np
from scipy.integrate import quad,nquad, cumtrapz
from scipy.special import legendre
from scipy.interpolate import interp1d
from matplotlib.pyplot import plot,show

class AdaptiveParticleManagement:
    def __init__(self):
        None
    def sample_unit_sphere(self, npoints, ndim=3):
        vec = np.random.randn(ndim, npoints)
        vec /= np.linalg.norm(vec, axis=0)
        return vec
    def sample_distribution(self,fL,V,Np):
        CDF = cumtrapz(V*V*fL[0,:],V,initial=0)
        sample = interp1d(CDF/CDF[-1],V)
        
        VD = np.trapz(V**3*fL[1,:],V)
        W = np.trapz(V*V*fL[0,:],V)
        
        r = np.random.random(Np)
        vs = sample(r)
        
#  Get uniform samples on unit sphere
        u = self.sample_unit_sphere(Np)
        
        ux = u[0,:]
        uy = u[1,:]
        uz = u[2,:]
            
        F0 = interp1d(V,V*V*fL[0,:])
        F1 = interp1d(V,V*V*fL[1,:])
        vd = vs*F1(vs)/F0(vs)
        v0 = np.sqrt(vs*vs-vd*vd*(1-uz*uz))-vd*uz
#  The thermal part of the isotropic part of the distribution
        vx = ux*v0
        vy = uy*v0
        vz = uz*v0
#  Ensure that the isotropic part has no net current    
        VX_avg = vx.sum()/vx.size
        VY_avg = vy.sum()/vy.size
        VZ_avg = vz.sum()/vz.size
        VD_avg = vd.sum()/vd.size
        
        vx = vx - VX_avg
        vy = vy - VY_avg
        vz = vz - VZ_avg
#  Ensure that the average drift of the sampled particles agrees with that of the original distribution
        vd = vd + VD - VD_avg
        vz = vz + vd
#  Set the new particle weights        
        w = W/Np
        return w,vx,vy,vz
        
    def set_positions(self):
        x = None
        y = None
        z = None
        return x,y,z

def Integrand(mu, L,v,VD,VT):
    pL = legendre(L)
    NORM = 1/(np.sqrt(2*np.pi)*VT**3)
    f = pL(mu)*np.exp( -0.5*(v*v-2*v*VD*mu+VD*VD)/VT**2 ) * NORM
    return f
    
def get_moments(fL,V):
# Density moment
    
    n0 = np.trapz(V**2*fL[0,:],V)
    print("Density",n0)
    #
    # Drift velocity Moment
    
    VD0 = np.trapz(V**3*fL[1,:],V)/n0
    
    print("Drift speed", VD0)
    #
    # Thermal speed
    
    Vsqrd = np.trapz(V**4*fL[0,:],V)/n0
    vth = np.sqrt(1/3*(Vsqrd-VD0**2))
    print("Thermal speed:", vth )
    return n0,VD,vth
#
#  Main program
    
V = np.linspace(0,5,501)
mu = np.linspace(-1,1,201)
VT = 1.0
VD = 0.4*VT

Lmax = 2
fL = np.zeros( (Lmax+1,V.size) )
#  Determine the expansion coefficients
for L in range(0,Lmax+1):
    F = np.zeros(0)
    for v in V:
        result = quad(Integrand, -1, 1, args=(L,v,VD,VT))[0]
        F = np.append(F,result)
    fL[L,:] = F

# Compute moments
#
get_moments(fL,V)

## Sample distribution
APM = AdaptiveParticleManagement()
Np = np.int(5E3)
w,vx,vy,vz = APM.sample_distribution(fL,V,Np)
v = np.sqrt(vx*vx+vy*vy+vz*vz)
print (v.size)
#
#Plots
f0,x = np.histogram(v,bins=50,range=[0,5],normed=True)
xx = 0.5*(x[1:]+x[0:-1])
plot(V,V*V*fL[0,:])
plot(xx,f0)
#show()
#
f1,x = np.histogram(v,bins=50,range=[0,5],density=True,normed=False,weights=vz/v)
norm = np.trapz(V*V*fL[1,:],V)
plot(V,V*V*fL[1,:])
plot(xx,norm*f1)

plot(V,V*V*fL[2,:])
show()
