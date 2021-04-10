from math import sqrt
from copy import deepcopy
class Fibre:
    def __init__(self, E1, E2, nu12, nu23, G12):
        self.E1=E1
        self.E2=E2
        self.E3=E2
        self.nu12=nu12
        self.nu13=nu12
        self.nu23=nu23
        self.G12=G12
        self.G13=G12
        self.G23=E2/2./(1+nu23)
        self.nu21=E2/E1*nu12
        self.nu31=self.nu21
        self.nu32=nu23

class Matrix:
    def __init__(self, E, nu):
        self.E=E
        self.nu=nu
        self.G=E/2./(1+nu)

def modifiedMixture(f, m, vf):
    print 'modofied mixture rule'
    E1=f.E1*vf+m.E*(1-vf)
    print 'E1=',E1
    svf=sqrt(vf)
    E2=1/(svf/(svf*f.E2+(1-svf)*m.E)+(1-svf)/m.E)
    print 'E2=E3=', E2
    G12=1/(svf/(svf*f.G12+(1-svf)*m.G)+(1-svf)/m.G)
    print 'G12=G13=', G12
    G23=1/(svf/(svf*f.G23+(1-svf)*m.G)+(1-svf)/m.G)
    print 'G23=', G23
    nu12=f.nu12*vf+m.nu*(1-vf)
    print 'nu12=nu13=', nu12
    nu23=E2/2./G23-1
    print 'nu23=', nu23
    return Fibre(E1,E2,nu12,nu23,G12)

def Chamis(f,m,vf):
    print 'model from Chamis paper'
    E1=f.E1*vf+m.E*(1-vf)
    print 'E1=',E1
    E2=m.E/(1-sqrt(vf)*(1-m.E/f.E2))
    print 'E2=',E2
    G12=m.G/(1-sqrt(vf)*(1-m.G/f.G12))
    print 'G12=G13=', G12
    G23=m.G/(1-sqrt(vf)*(1-m.G/f.G23))
    print 'G23=', G23
    nu12=f.nu12*vf+m.nu*(1-vf)
    print 'nu12=nu13=', nu12
    nu23=E2/2./G23-1
    print 'nu23=', nu23
    return Fibre(E1,E2,nu12,nu23,G12)

def Pochiraju(f,m,vf):
    print 'model from Pochiraju paper'
    vm=1-vf
    kf=f.E1*f.E2*0.5/((1.-f.nu23)*f.E1-2.*f.nu12*f.nu12*f.E2)
    km=0.5*m.E/(1-m.nu-2*m.nu*m.nu)
    tmp=((kf+m.G)*km+(kf-km)*m.G*vf)
    tmp2=m.G*vm*vf
    E1=f.E1*vf+m.E*vm+4.*(m.nu-f.nu12)**2*kf*km*tmp2/tmp
    nu12=f.nu12*vf+m.nu*vm+(m.nu-f.nu12)*(km-kf)*tmp2/tmp
    tmp3=f.G12+m.G
    tmp4=(f.G12-m.G)*vf
    G12=m.G*(tmp3+tmp4)/(tmp3-tmp4)
    tmp5=f.G23+m.G
    tmp6=(f.G23-m.G)*vf
    G23=m.G*(km*tmp5+2*f.G23*m.G+km*tmp6)/(km*tmp5+2*f.G23*m.G-(km+2*m.G)*tmp6)
    tmp7=kf+m.G
    tmp8=(kf-km)*vf
    kt=(tmp7*km+tmp8*m.G)/(tmp7-tmp8)
    E2=1/(0.25/kt+0.25/G23+nu12*nu12/E1)
    nu23=0.5*(2*E1*kt-E1*E2-4*nu12*nu12*kt*E2)/E1/kt
    print 'E1=', E1
    print 'E2=E3=', E2
    print 'G12=G13=', G12
    print 'G23=', G23
    print 'nu12=nu13=',nu12
    print 'nu23=',nu23
    return Fibre(E1,E2,nu12,nu23,G12)

def D(o):
    Q=[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
    E1=o.E1
    E2=o.E2
    E3=o.E3
    G12=o.G12
    G23=o.G23
    G13=o.G13
    nu12=o.nu12
    nu13=o.nu13
    nu23=o.nu23
    nu21=o.nu21
    nu32=o.nu32
    nu31=o.nu31
    delta=1-nu12*nu21-nu23*nu32-2*nu21*nu32*nu13
    Q[0][0]=(1-nu23*nu32)*E1/delta
    Q[0][1]=Q[1][0]=(nu31+nu21*nu23)*E1/delta
    Q[0][2]=Q[2][0]=(nu31+nu21*nu32)*E1/delta
    Q[1][1]=(1-nu13*nu31)*E2/delta
    Q[2][2]=(1-nu12*nu21)*E3/delta
    Q[1][2]=Q[2][1]=(nu32+nu12*nu31)*E2/delta
    Q[3][3]=G12
    Q[4][4]=G13
    Q[5][5]=G23
    return Q

def S(o):
    Q=[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
    E1=o.E1
    E2=o.E2
    E3=o.E3
    G12=o.G12
    G23=o.G23
    G13=o.G13
    nu12=o.nu12
    nu13=o.nu13
    nu23=o.nu23
    nu21=o.nu21
    nu32=o.nu32
    nu31=o.nu31
    Q[0][0]=1./E1
    Q[1][1]=1./E2
    Q[2][2]=1./E3
    Q[3][3]=1./G12
    Q[4][4]=1./G13
    Q[5][5]=1./G23
    Q[0][1]=Q[1][0]=-nu12/E1
    Q[0][2]=Q[2][0]=-nu13/E1
    Q[1][2]=Q[2][1]=-nu23/E2
    Q[0][1]=Q[1][0]=-nu12/E1
    return Q

def matprint(M):
        for l in M:
                print ''.join(['%15.4e' % d for d in l])

def invM(CC):
        C=[list(CCC) for CCC in CC]
        S=[]
        for i in xrange(6):
                S.append([0.0,0.0,0.0,0.0,0.0,0.0])
        for i in xrange(6):
                S[i][i]=1.0
        for i in xrange(6):
                tmp=C[i][i]
                for j in xrange(6):
                        C[i][j]/=tmp
                        S[i][j]/=tmp
                for k in xrange(i+1,6):
                        tmp=C[k][i]
                        for j in xrange(6):
                                C[k][j]-=C[i][j]*tmp
                                S[k][j]-=S[i][j]*tmp

        for i in xrange(5,-1,-1):
                for k in xrange(i-1,-1,-1):
                        tmp=C[k][i]
                        for j in xrange(6):
                                C[k][j]-=C[i][j]*tmp
                                S[k][j]-=S[i][j]*tmp
        return S

m=Matrix(3.1,0.42)
f1=Fibre(127,3.35,0.26,0.17,3)
f2=Fibre(226,12.9,0.31,0.2,60)
modifiedMixture(f2,m,0.4)
#modifiedMixture(f2,m,0.4)
Chamis(f2,m,0.4)
#Chamis(f2,m,0.4)
c1=Pochiraju(f2,m,0.4)
print 'D:'
DD=D(c1)
matprint(DD)
SS=S(c1)
print 'S:'
matprint(SS)
print 'S-1:'
matprint(invM(SS))
print 'D-1:'
matprint(invM(DD))


