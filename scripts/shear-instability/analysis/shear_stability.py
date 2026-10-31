"""Full-depth, frozen-profile CL roll stability with homogeneous stress BCs.

x-independent perturbations use u=cos(m*pi*(z+H)/H), w=sin(m*pi*(z+H)/H).
Thus u_z=v_z=0 and w=0 at both walls. v follows incompressibility. The
prescribed wind/maintenance has no perturbation, so these are the correct
linear perturbation boundary conditions even when the base stress is nonzero.
Frozen eigenvalues and their integrals are diagnostics, not a nonautonomous
propagator or a proof that an evolving turbulent flow follows one eigenmode.
"""
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=32)
def basis(h,n,quadrature):
    z=-h+(np.arange(quadrature)+.5)*h/quadrature
    u=np.cos(np.pi*(z[:,None]+h)*np.arange(n+1)[None,:]/h)*np.sqrt(2/h)
    u[:,0]/=np.sqrt(2)
    w=np.sin(np.pi*(z[:,None]+h)*np.arange(1,n+1)[None,:]/h)*np.sqrt(2/h)
    return z,u,w


def operator(z,q,stokes_shear,ky,h,nu,n=48,quadrature=384,boundary_q=None):
    """Galerkin generator in the Euclidean kinetic-energy coordinates.

    u_t = nu*(Dzz-ky^2)u - Q*w
    (Dzz-ky^2)w_t = nu*(Dzz-ky^2)^2*w + ky^2*Us_z*u.
    The w energy weight is 1+(m*pi/H/ky)^2, accounting for transverse v.
    """
    if ky<=0 or h<=0 or nu<0 or n<2 or quadrature<2*n+2:raise ValueError('Invalid stability grid')
    z=np.asarray(z);q=np.asarray(q);ss=np.asarray(stokes_shear)
    if len(z)!=len(q) or len(z)!=len(ss) or not np.all(np.diff(z)>0):raise ValueError('Invalid profile')
    zz,u,w=basis(float(h),int(n),int(quadrature))
    bounds=(q[0],q[-1]) if boundary_q is None else boundary_q
    qi=np.interp(zz,z,q,left=bounds[0],right=bounds[1]);si=np.interp(zz,z,ss)
    cu=(u.T*(qi*h/quadrature))@w;cs=(w.T*(si*h/quadrature))@u
    mu=np.pi*np.arange(n+1)/h;mw=np.pi*np.arange(1,n+1)/h
    ku=ky**2+mu**2;kw=ky**2+mw**2;weight=np.sqrt(kw)/ky
    a=np.zeros((2*n+1,2*n+1));a[:n+1,:n+1]=np.diag(-nu*ku);a[n+1:,n+1:]=np.diag(-nu*kw)
    a[:n+1,n+1:]=-cu/weight[None,:]
    a[n+1:,:n+1]=-(weight*ky**2/kw)[:,None]*cs
    return a


def leading_rate(z,q,ss,ky,h,nu,n=48,quadrature=384,boundary_q=None):
    eig=np.linalg.eigvals(operator(z,q,ss,ky,h,nu,n,quadrature,boundary_q))
    winner=eig[np.argmax(eig.real)]
    return float(winner.real),float(abs(winner.imag))


def native_shear(ue,h,k,stokes_speed,alpha,stokes_reference,ustar,nu):
    """Native staggered vertical curl and gradient of a horizontal mean."""
    ue=np.asarray(ue);dz=h/len(ue)
    qbottom=-alpha*2*k*stokes_reference*np.exp(-2*k*h)
    qtop=-alpha*2*k*stokes_reference+ustar**2/nu
    # S may differ from reference only in controlled wave-removal tests. This
    # pilot fixes S=reference, checked by caller, and retains baseline BCs.
    faces=np.r_[qbottom,np.diff(ue)/dz,qtop]
    q=.5*(faces[:-1]+faces[1:]);beta=np.diff(faces)/dz
    return q,beta,(float(qbottom),float(qtop))


def sampled_spectrum(z,q,ss,ly,h,nu,boundary_q=None,n=48,modes=tuple(range(1,32)),reference_ny=64):
    records=[]
    for mode in modes:
        growth,freq=leading_rate(z,q,ss,2*np.pi*mode/ly,h,nu,n=n,boundary_q=boundary_q)
        records.append(dict(ky_mode=int(mode),growth_s_inverse=growth,frequency_s_inverse=freq))
    winner=max(records,key=lambda r:r['growth_s_inverse'])
    # The continuum CL spectrum can prefer wavelengths poorly represented by
    # the native 64-cell transverse grid. Search beyond its admissible band as
    # an explicit resolution diagnostic; never call those simulated modes.
    outside=[]
    for mode in (32,48,64):
        g,f=leading_rate(z,q,ss,2*np.pi*mode/ly,h,nu,n=n,boundary_q=boundary_q)
        outside.append(dict(ky_mode=mode,growth_s_inverse=g,frequency_s_inverse=f))
    refined,_=leading_rate(z,q,ss,2*np.pi*winner['ky_mode']/ly,h,nu,n=72,boundary_q=boundary_q)
    coarse,_=leading_rate(z,q,ss,2*np.pi*winner['ky_mode']/ly,h,nu,n=24,boundary_q=boundary_q)
    error=abs(refined-winner['growth_s_inverse'])/max(abs(refined),1e-4)
    four=max((r for r in records if reference_ny/r['ky_mode']>=4),key=lambda r:r['growth_s_inverse'])
    six=max((r for r in records if reference_ny/r['ky_mode']>=6),key=lambda r:r['growth_s_inverse'])
    return dict(tested_modes=records,winning_mode=winner['ky_mode'],growth_s_inverse=winner['growth_s_inverse'],
                refined_growth_s_inverse=float(refined),coarse_growth_s_inverse=float(coarse),vertical_relative_difference=float(error),
                vertical_converged=bool(error<=.05),horizontal_edge=bool(winner['ky_mode']==max(modes)),
                wavelength_m=float(ly/winner['ky_mode']),reference_cells_per_wavelength=float(reference_ny/winner['ky_mode']),
                horizontal_grid_resolved=bool(reference_ny/winner['ky_mode']>=4),outside_native_band=outside,
                four_cell_band_growth_s_inverse=four['growth_s_inverse'],four_cell_band_winning_mode=four['ky_mode'],
                six_cell_band_growth_s_inverse=six['growth_s_inverse'],six_cell_band_winning_mode=six['ky_mode'],
                instability_only_in_subfourcell_band=bool(winner['growth_s_inverse']>0 and four['growth_s_inverse']<=0),
                faster_outside_native_band=bool(max(x['growth_s_inverse'] for x in outside)>winner['growth_s_inverse']+1e-6),
                interpretation='Continuum frozen-profile spectrum: every spanwise mode 1–31 is tested, with 32/48/64 beyond-native-band checks. Vertical convergence compares 48 and 72 basis modes. Fewer than four reference-grid cells per wavelength is unresolved; four cells is only a screening floor, not convergence. This operator does not replace the native centered-advection symbol by a continuum-validated discretization.')
