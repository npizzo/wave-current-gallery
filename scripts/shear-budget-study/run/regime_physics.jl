# Controlled background maintenance. Only this new campaign uses this force.
# The constant ScalarDiffusivity acts on prognostic Lagrangian velocity.
# Its native vertical stencil, including prescribed gradient boundary fluxes,
# is cancelled on the prescribed background, leaving explicit diffusion of
# perturbations. This is not an unforced laboratory or molecular-flow model.
@inline function maintained_background_forcing(i,j,k,grid,clock,fields,p)
    b=(p.waves-p.alpha)*p.S
    z=-p.h+(k-.5)*p.dz
    center=b*exp(2p.k*z)
    lap=if k==1
        (b*exp(2p.k*(z+p.dz))-center-2p.k*b*exp(-2p.k*p.h)*p.dz)/p.dz^2
    elseif k==p.nz
        (2p.k*b*p.dz-center+b*exp(2p.k*(z-p.dz)))/p.dz^2
    else
        (b*exp(2p.k*(z+p.dz))-2center+b*exp(2p.k*(z-p.dz)))/p.dz^2
    end
    return -p.nu*lap
end

function regime_forcing_parameters(;waves,alpha,S,k,h,nz,nu)
    (;waves,alpha,S,k,h,nz,nu,dz=h/nz)
end
