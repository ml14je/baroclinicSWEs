#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:32:50 2022

@author: amtsdg
"""

import matplotlib.pyplot as pt, numpy as np

def test_func(x):
    # just a test function for differentiation
    
    L=1e5
    f=np.sin(np.pi*x/L-1/2)
    fx=(np.pi/L)*np.cos(np.pi*x/L-1/2)
    
    return f, fx

def calc_csq(h,gr,d):
    
    csq=gr*d*(1-d/h)
    csqh=gr*d**2/h**2
    csqhh=-2*gr*d**2/h**3
    
    return csq, csqh, csqhh

def calc_h(x,hL,hR,lx):
    
    # hL: depth on left
    # hR: depth on right
    # lx: width of topo
    
    h=hL+(x>0)*(x<lx)*(hR-hL)*np.sin(0.5*np.pi*x/lx)**2+(hR-hL)*(x>=lx)
    hx=(x>0)*(x<lx)*(hR-hL)*np.sin(np.pi*x/lx)*(0.5*np.pi/lx)
    
    return h, hx

def calc_Hm(h,gr,d):

    csq,csqh,csqhh=calc_csq(h,gr,d)
    Hm=csq/csqh
    
    return Hm    

def calc_U(x,hL,hR,lx,Q):
    
    h=calc_h(x,hL,hR,lx)[0]
    
    U=Q/h # could be complex, if desired
    
    return U

def Z1(h,gr,d):
    csq,csqh,csqhh=calc_csq(h,gr,d)
    Z=np.sqrt(csqh)

    return Z

def modal_interaction_coefficent(h, gr, d):
    g = 9.81
    c1sq,c1sqh,c1sqhh = calc_csq(h,gr,d)
    c0sq,c0sqh = g * h, g
    
    T = (np.sqrt(c1sqh) * np.sqrt(c0sqh))/(c0sq)
    
    return T
    

lx=50e3 # slope width (m)
hL=200  # shelf depth (m)
hR=4000 # deep ocean depth (m)

d=150  # upper-layer depth (m)
g = 9.81 # gravity (m/s^2)
gr = g * (10/1036) # reduced gravity (m/s^2)
f = 1e-4 # Coriolis parameter
omega=1.4e-4 # diurnal forcing frequency (s^-1)
Q=10       # volume flux for barotropic tide (m^2/s)
rho0=1036  # density (kg/m^3)

nx=1000

# set up grids
dx=lx/nx
xu=np.linspace(0,lx,nx+1)
xu=xu.reshape(-1,1)
xp=xu[0:-1]+0.5*dx
nux=nx+1
npx=nx

# make some interpolation matrices
# From u grid to p grid:
Iup=np.eye(nx,nx+1,0)+np.eye(nx,nx+1,1)
Iup=Iup/2
# From p grid to u grid:
Ipu=np.eye(nx+1,nx,-1)+np.eye(nx+1,nx,0)
Ipu=Ipu/2
# Fix endpoints (although not needed):
Ipu[0,0:2]=np.array([1.5,-0.5])
Ipu[nx,-2:]=np.array([-0.5,1.5])

# make some differentiation matrices
# From u grid to pgrid:
Dup=np.eye(nx,nx+1,1)-np.eye(nx,nx+1,0)
Dup=Dup/dx
# From p grid to ugrid:
Dpu=np.eye(nx+1,nx,0)-np.eye(nx+1,nx,-1)
Dpu=Dpu/dx
# Put first-order term at ends (although not needed):
Dpu[0,0:2]=np.array([-1,1])/dx
Dpu[nx,-2:]=np.array([-1,1])/dx


# now calculate quantities for IT calculation. 
# First on the p grid:  
hp,hxp=calc_h(xp,hL,hR,lx)
pt.plot(xp*1e-3, hxp)
pt.show()
Tp=modal_interaction_coefficent(hp, gr, d)
# now on the u grid: 
hu,hxu=calc_h(xu,hL,hR,lx)
csqu,csqhu,csqhhu=calc_csq(hu,gr,d)
Tu=modal_interaction_coefficent(hu, gr, d)

# barotropic tide (on p grid)
Up=calc_U(xp,hL,hR,lx,Q)

u=np.zeros([nx+1,0])
p=np.zeros([nx,0])

# indices for u and p halves of (u,p) vectors:
iu=np.s_[0:nx+1]
ip=np.s_[nx+1:2*nx+2]
# indices for u and p quarters of (u,p) matrices:
iuu=np.s_[iu,iu]
iup=np.s_[iu,ip]
ipu=np.s_[ip,iu]
ipp=np.s_[ip,ip]

# make (u,p) matrix system
mat=np.zeros([2*nx+1,2*nx+1],dtype='complex')
# u equation:
mat[iuu]=-1j*omega*(1 - (f/omega)**2)*np.eye(nx+1,nx+1)
mat[iup]=Dpu

# p equation:
mat[ipp]=-1j*omega*(1 - (f/omega)**2)*np.eye(nx,nx)
temp1=csqu[:,0]
mat[ipu]=(Dup @ np.diag(temp1))

# RHS (forcing):
Z0 = np.sqrt(g)
c0squ, c0sqp = g * hu, g * hp
p = 0 # pressure is 1j * d_x(Q)/w = 0
rhs=np.zeros([2*nx+1,1],dtype='complex')
rhs[iu] = -Tu * hxu * p / Z0
rhs[ip] = -Tp * hxp * Up * c0sqp / Z0

# Radiating bcs:
cL=np.sqrt(calc_csq(hL,gr,d)[0])
cR=np.sqrt(calc_csq(hR,gr,d)[0])
# set cL u + p = 0  at left:
mat[0,:]=0
mat[0,0]=+cL*np.sqrt((1 - (f/omega)**2))
mat[0,nx+1]=1
# set cR u - p = 0 at right:
mat[nx,:]=0
mat[nx,nx]=-cR*np.sqrt((1 - (f/omega)**2))
mat[nx,2*nx]=1
# remove forcing at boundary: 
rhs[0]=0
rhs[nx]=0

# Solve: 
up=np.linalg.solve(mat,rhs)

# extract u and p:
u=up[iu] * Z1(hu,gr,d)
p=up[ip] * Z1(hp,gr,d)

pt.figure(3)
pt.clf()
pt.subplot(2,1,1)
pt.plot(xu*1e-3,u.real*1e2,'r-')
pt.plot(xu*1e-3,u.imag*1e2,'b-')
pt.xlabel(r'$x$ (km)')
pt.ylabel(r'$u$ (cm/s)')
pt.grid()
pt.subplot(2,1,2)
pt.plot(xp*1e-3,rho0*p.real,'r-')
pt.plot(xp*1e-3,rho0*p.imag,'b-')
pt.xlabel(r'$x$ (km)')
pt.ylabel(r'$p$ ($\rm{kgm^2/s^2}$)')
pt.grid()

# energy flux calculation:
HL=calc_Hm(hL,gr,d)
HR=calc_Hm(hR,gr,d)
JL=0.5*HL*rho0*(u[0]*np.conj(p[0])).real[0]
JR=0.5*HR*rho0*(u[-1]*np.conj(p[-1])).real[0]

print(f'JL: {JL:.2f} W/m') #Use of f-string and format
print(f'JR: {JR:.2f} W/m') #Use of f-string and format
print(f'|JR|+|JL|: {(JR-JL):.2f} W/m')

drag=-rho0*p*hxp

pt.figure(4)
pt.clf()
pt.plot(1e-3*xp, 1e3 * 0.5*(drag*Up.conjugate()).real)
pt.xlabel(r'$x$ (km)')
pt.ylabel(r'Drag ($\rm{mW/m^2}$)')
pt.grid()

# Calculate dissipation directly: 
diss=0.5*np.sum((drag*np.conj(Up)).real)*dx
print(f'D: {diss:.2f} W/m.')