import numpy as np
from scipy.integrate import quad,nquad
from scipy import integrate
from scipy.special import legendre
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from scipy.optimize import fsolve
from matplotlib.pyplot import plot,show

class AdaptiveParticleManagement:
    def __init__(self,Np, rho,J, particles):
        self.Np = Np
        self.rho = rho
        self.J = J
        self.VD = J/rho
        self.particles = particles
        
    def sample_distribution_bak(self,fL,V):
        phi = np.linspace(0,2.*np.pi,31)
        phi = 0.5*(phi[1:]+phi[0:-1])
        Nv = 30
        Nmu = 30
        Nphi = phi.size
        Np = Nv*Nmu*Nphi
        print(Np)
        FL = V*V*fL
        n = np.trapz(FL,V)
        W = np.ones(Np)*n[0]/Np
        vx = np.zeros(0)
        vy = np.zeros(0)
        vz = np.zeros(0)
#        p_v = 0.5*FL[0:,np.newaxis,np.newaxis]+1.5*uz*FL[1,:,np.newaxis,np.newaxis]
        mu = np.linspace(-1,1,201)
        P_mu = 0.5*n[0]+1.5*n[1]*mu
        CDF_mu = cumtrapz(P_mu,mu,initial=0)
        sample_P_mu = interp1d(CDF_mu/CDF_mu[-1],mu)
        r = np.random.random(Nv*Nmu)
        mu_s = sample_P_mu(r)
        COS = mu_s
        SIN = np.sqrt(1-COS*COS)
        for PHI in phi:
            for mu in mu_s:
                F = FL[0,:]+3*mu*FL[1,:]
                CDF = cumtrapz(F,V,initial=0)
                sample_F = interp1d(CDF/CDF[-1],V)
                v = sample_F(r)
            vx = np.append(vx,v*SIN*np.cos(PHI))
            vy = np.append(vy,v*SIN*np.sin(PHI))
            vz = np.append(vz,v*COS)

        vx_AVG = np.sum(vx)/Np
        vx = vx - vx_AVG
        vy_AVG = np.sum(vy)/Np
        vy = vy - vy_AVG
        vz_AVG = np.sum(vz)/Np
        vz = vz - vz_AVG + self.VD     
        
        print(vx.shape)
        print(vy.shape)
        print(vz.shape)
        return vx,vy,vz
        
    def sample_distribution(self,fL,V,mu):
        Np = self.Np
        L = np.size(fL[:,0])
        FL = V*V*fL
        n = np.trapz(FL,V)

        W = np.ones(Np)*n[0]/Np
        p_mu = 0
        for l in range(L):
            Pl = legendre(l)
            p_mu = p_mu + 0.5*(2*l+1)*Pl(mu)*n[l]/n[0]
        self.p_mu = p_mu
        CDF_mu = cumtrapz(p_mu,mu,initial=0)
        sample_mu = interp1d(CDF_mu/CDF_mu[-1],mu)
        r = np.random.random(Np)
        mu_s = sample_mu(r)
        phi_s = 2*np.pi*np.random.random(Np)

        v_s = np.zeros(Np)
        P_mu = interp1d(mu,p_mu)
        p_mu_s = P_mu(mu_s)
        for i in range(Np):
            p_v = 0.0
            for l in range(L):
                Pl = legendre(l)
                p_v = p_v + 0.5*(2*l+1)*Pl(mu_s[i])*FL[l,:]
            p_v = p_v/p_mu_s[i]
            CDF_v = cumtrapz(p_v,V,initial=0)
            sample_v = interp1d(CDF_v/CDF_v[-1],V)
            r = np.random.random()
            v_s[i] = sample_v(r)
        
        SIN = np.sqrt(1-mu_s*mu_s)
        vx = v_s*np.cos(phi_s)*SIN
        vy = v_s*np.sin(phi_s)*SIN
        vz = v_s*mu_s
#  Subtract off the known mean of the existing distribution        
        vx_AVG = np.sum(vx)/Np
        vx = vx - vx_AVG
        vy_AVG = np.sum(vy)/Np
        vy = vy - vy_AVG
        vz_AVG = np.sum(vz)/Np
        vz = vz - vz_AVG + self.VD     

        return vx,vy,vz
        
    def rotate_v(self,particles):
        None
        
    def get_distribution(self,V,w):
        dim = V[0,:].size
        # Volume = 1.0
        H,edges = np.histogramdd(V,bins = 50,normed=True,weights=w)
        # for d in range(dim):
        #     dx = (edges[d][1]-edges[d][0])
        #     Volume = Volume*(dx)

    # Turn histogram into distribution by dividing by volume
        # f = H/Volume
        
        F ={"bins":edges,"distribution":H}
        return F
        
    def get_moments(self, vx,vy,vz):
        Np = particles.vx.size

        avg_vx = np.sum(vx)/Np
        avg_vy = np.sum(vy)/Np
        avg_vz = np.sum(vz)/Np
        speed = np.sqrt(vx*vx+vy*vy+vz*vz)
        avg_speed = np.sum(speed)/Np
        return avg_vx,avg_vy,avg_vz, avg_speed
        
    def update(self):
        vx = particles.vx 
        vx = particles.vy 
        vx = particles.vz 
        self.VX,self.VY,self.VZ = get_moments(vx,vy,vz)
        theta = self.VZ
        self.rotate(X,theta,phi)
        vx,vy,vz = self.get_distribution()
        

        None
        
    
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

    

    
#  Compute Drifting Maxwellian and multi-expansion distributions

V = np.linspace(0,4,501)
mu = np.linspace(-1,1,21)
VT = 1.0
VD = 0.2*VT

Lmax = 1
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

## Sample
Nv = 51
Nmu = 10
Nphi = 10
Np = Nv*Nmu*Nphi
mu = np.linspace(-1,1,Nmu+1)
mu = 0.5*(mu[1:]+mu[0:-1])
phi = np.linspace(0,2*np.pi,Nphi+1)
phi = 0.5*(phi[1:]+phi[0:-1])
vx = np.zeros(0)
vy = np.zeros(0)
vz = np.zeros(0)
CDF = cumtrapz(V**2*fL[0,:],V,initial=0)
sample_f0 = interp1d(CDF/CDF[-1],V)
for COS in mu:
    SIN = np.sqrt(1-COS*COS)
    for PHI in phi:
        r = np.random.random(Nv)
        vsqrd = sample_f0(r)
        v0 = np.sqrt((vsqrd-VD**2)/3.0)
        vx = np.append(vx,v0*SIN*np.cos(PHI))
        vy = np.append(vy,v0*SIN*np.sin(PHI))
        vz = np.append(vz,v0*COS)
        
print("<vx>:",np.sum(vx)/vx.size)
print("<vy>:",np.sum(vy)/vx.size)
print("<vz>:",np.sum(vz)/vx.size)
v = np.sqrt(vx*vx+vy*vy+vz*vz)
print("<v^2>:",np.sum(v*v)/v.size)


##Sample
Np = 50000
rho = 1.0
J = rho*VD
particles = 1.0
APM = AdaptiveParticleManagement(Np, rho,J, particles)
vx,vy,vz = APM.sample_distribution(fL,V,mu)
wgt = np.ones(Np)
##
VXY = np.array([vx,vy]).T
Fxy = APM.get_distribution(VXY,weights)
np.savetxt("N:\\Codes\\turbopy\\chemistry\\Notes\FXY.txt",Fxy["distribution"])
VXZ = np.array([vx,vz]).T
Fxz = APM.get_distribution(VXZ,weights)
np.savetxt("N:\\Codes\\turbopy\\chemistry\\Notes\FXZ.txt",Fxz["distribution"])
#
#  Subtract off averages to that the sample average is the same of the original distribution


##plots
v = np.linspace(-4,4,201)
fx = np.exp(-0.5*v*v/VT**2)
fx = fx/np.trapz(fx,v)
hist_x ,x = np.histogram(vx,bins=30,range=[-4,4],density=True)
A = zip(x,hist_x)
B = np.array(list(A))
np.savetxt("N:\\Codes\\turbopy\\chemistry\\Notes\H_x.txt",B)
x = 0.5*(x[1:]+x[0:-1])
plot(v,fx)
plot(x,hist_x)
hist_y ,y = np.histogram(vy,bins=30,range=[-4,4],density=True)
A = zip(y,hist_y)
B = np.array(list(A))
np.savetxt("N:\\Codes\\turbopy\\chemistry\\Notes\\H_y.txt",B)
y = 0.5*(y[1:]+y[0:-1])
plot(y,hist_y)
show()

##
fz = np.exp(-0.5*(v-VD)**2/VT**2)
fz = fz/np.trapz(fz,v)
hist_z ,z = np.histogram(vz,bins=30,range=[-4,4],density=True)
A = zip(z,hist_z)
B = np.array(list(A))
np.savetxt("N:\\Codes\\turbopy\\chemistry\\Notes\\H_z.txt",B)
z = 0.5*(z[1:]+z[0:-1])
plot(v,fz)
plot(z,hist_z)
show()
