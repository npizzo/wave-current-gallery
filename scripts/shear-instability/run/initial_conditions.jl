using Random, Statistics

"Native periodic-x/y, bounded-z divergence; all velocity arrays retain staggering."
function native_divergence(u,v,w,dx,dy,dz)
    (circshift(u,(-1,0,0)).-u)./dx .+
        (size(v,2)==1 ? zero(v) : (circshift(v,(0,-1,0)).-v)./dy) .+
        (w[:,:,2:end].-w[:,:,1:end-1])./dz
end

function center_velocities(u,v,w)
    uc=(u.+circshift(u,(-1,0,0)))./2
    vc=size(v,2)==1 ? copy(v) : (v.+circshift(v,(0,-1,0)))./2
    wc=(w[:,:,1:end-1].+w[:,:,2:end])./2
    return uc,vc,wc
end

function velocity_norms(u,v,w)
    uc,vc,wc=center_velocities(u,v,w)
    (;inplane=sqrt(mean(abs2,uc)+mean(abs2,wc)),
      total=sqrt(mean(abs2,uc)+mean(abs2,vc)+mean(abs2,wc)),
      u=sqrt(mean(abs2,uc)),v=sqrt(mean(abs2,vc)),w=sqrt(mean(abs2,wc)))
end

"RMS using cell-averaged native face squares, consistent with the runtime energy."
function native_velocity_rms(u,v,w)
    sqrt(mean(abs2,u)+mean(abs2,v)+mean((w[:,:,1:end-1].^2 .+ w[:,:,2:end].^2)./2))
end

function curl_potential(ax,ay,az,dx,dy,dz)
    u=(circshift(az,(0,-1,0)).-az)./dy .- (ay[:,:,2:end].-ay[:,:,1:end-1])./dz
    v=(ax[:,:,2:end].-ax[:,:,1:end-1])./dz .- (circshift(az,(-1,0,0)).-az)./dx
    w=(circshift(ay,(-1,0,0)).-ay)./dx .- (circshift(ax,(0,-1,0)).-ax)./dy
    return u,v,w
end

"""
Generate an exactly y-invariant discrete streamfunction curl. The eddy field
has no horizontally averaged u profile and zero normal velocity at both walls.
Its prescribed RMS is sqrt(<u_c²+w_c²>), using centered native velocities.
"""
function embedded_eddy(nx,ny,nz,lx,ly,h,seed,wavelength,rms)
    dx,dz=lx/nx,h/nz;xf=collect(0:nx-1).*dx;zf=collect(0:nz).*dz
    psi=zeros(nx,nz+1);rng=Xoshiro(seed);ke=2pi/wavelength
    mxmax=min(nx÷6,ceil(Int,1.6ke*lx/(2pi)))
    nzmax=min(nz÷6,ceil(Int,1.6ke*h/pi))
    for mx in 1:mxmax,n in 1:nzmax
        kh=2pi*mx/lx;kv=pi*n/h;ratio=hypot(kh,kv)/ke
        .55<=ratio<=1.6 || continue
        weight=exp(-((ratio-1)/.28)^2)*randn(rng)/(kh^2+kv^2)
        phase=2pi*rand(rng)
        psi .+= weight .* cos.(kh.*xf.+phase) .* transpose(sin.(kv.*zf))
    end
    psi[:,1].=0;psi[:,end].=0
    u2=diff(psi;dims=2)./dz
    w2=-(circshift(psi,(-1,0)).-psi)./dx
    # Removing the x-constant part of u does not change its native divergence.
    u2 .-= mean(u2;dims=1)
    u=repeat(reshape(u2,nx,1,nz),1,ny,1)
    w=repeat(reshape(w2,nx,1,nz+1),1,ny,1)
    v=zeros(nx,ny,nz)
    norm=velocity_norms(u,v,w).inplane
    norm>0 || error("Grid cannot represent requested injection band")
    u.*=rms/norm;w.*=rms/norm
    return u,v,w
end

"""
Three-component discrete curl with only nonzero y Fourier modes. Potentials
Ax,Ay vanish on z walls; Az has zero normal derivative there. Native mixed
differences commute, so divergence cancels to roundoff without projection.
The in-plane injection band is held fixed while transverse wavelengths vary.
"""
function transverse_eddy(nx,ny,nz,lx,ly,h,seed,wavelength)
    ny>=8 || error("Three-dimensional perturbations require NY>=8")
    dx,dy,dz=lx/nx,ly/ny,h/nz
    xf=collect(0:nx-1).*dx;xc=xf.+dx/2
    yf=collect(0:ny-1).*dy;yc=yf.+dy/2
    zf=collect(0:nz).*dz;zc=zf[1:end-1].+dz/2
    ax=zeros(nx,ny,nz+1);ay=similar(ax);fill!(ay,0)
    az=zeros(nx,ny,nz)
    rng=Xoshiro(seed+104729);ke=2pi/wavelength
    modes=Tuple{Int,Int,Int}[]
    mxmax=min(nx÷6,ceil(Int,1.6ke*lx/(2pi)))
    nzmax=min(nz÷6,ceil(Int,1.6ke*h/pi))
    mymax=max(1,min(ny÷6,ceil(Int,ke*ly/(2pi))))
    for mx in -mxmax:mxmax,n in 1:nzmax
        ratio=hypot(2pi*mx/lx,pi*n/h)/ke
        .55<=ratio<=1.6 || continue
        push!(modes,(mx,rand(rng,1:mymax),n))
    end
    shuffle!(rng,modes);resize!(modes,min(length(modes),48))
    isempty(modes) && error("Grid cannot represent transverse injection band")
    for (mx,my,n) in modes
        kh,ky,kv=2pi*mx/lx,2pi*my/ly,pi*n/h
        ratio=hypot(kh,kv)/ke
        weight=exp(-((ratio-1)/.28)^2)/(kh^2+ky^2+kv^2)
        axxy=randn(rng).*cos.(kh.*xc .+ transpose(ky.*yf) .+2pi*rand(rng))
        ayxy=randn(rng).*cos.(kh.*xf .+ transpose(ky.*yc) .+2pi*rand(rng))
        azxy=randn(rng).*cos.(kh.*xf .+ transpose(ky.*yf) .+2pi*rand(rng))
        ax .+= weight.*reshape(axxy,nx,ny,1).*reshape(sin.(kv.*zf),1,1,nz+1)
        ay .+= weight.*reshape(ayxy,nx,ny,1).*reshape(sin.(kv.*zf),1,1,nz+1)
        az .+= weight.*reshape(azxy,nx,ny,1).*reshape(cos.(kv.*zc),1,1,nz)
    end
    ax[:,:,1].=0;ax[:,:,end].=0;ay[:,:,1].=0;ay[:,:,end].=0
    ax .-= mean(ax;dims=2);ay .-= mean(ay;dims=2);az .-= mean(az;dims=2)
    curl_potential(ax,ay,az,dx,dy,dz)
end

function initial_velocity_fields(;nx,ny,nz,lx,ly,h,k,S_reference,waves,alpha,uoffset,
    eddy_rms,injection_wavelength,perturb_energy_ratio,seed,ic_kind="embedded")
    ic_kind in ("embedded","three_d") || error("IC_KIND must be embedded or three_d")
    zc=[-h+(j-.5)*h/nz for j in 1:nz]
    usref=S_reference.*exp.(2k.*zc)
    background=(waves-alpha).*usref.+uoffset
    if ic_kind=="embedded"
        ub,vb,wb=embedded_eddy(nx,ny,nz,lx,ly,h,seed,injection_wavelength,eddy_rms)
    else
        ub=zeros(nx,ny,nz);vb=copy(ub);wb=zeros(nx,ny,nz+1)
    end
    up=zeros(nx,ny,nz);vp=copy(up);wp=zeros(nx,ny,nz+1)
    perturbation_shape=nothing
    if perturb_energy_ratio>0 || ic_kind=="three_d"
        up,vp,wp=transverse_eddy(nx,ny,nz,lx,ly,h,seed,injection_wavelength)
        perturbation_shape=(copy(up),copy(vp),copy(wp))
        r=velocity_norms(up,vp,wp)
        scale=ic_kind=="three_d" ? eddy_rms/r.inplane : sqrt(perturb_energy_ratio)*eddy_rms/r.total
        up.*=scale;vp.*=scale;wp.*=scale
    end
    # Retain separated fields to fingerprint the identical baseline and added
    # transverse perturbation across plus/minus probes and robustness members.
    u=ub.+up.+reshape(background,1,1,nz);v=vb.+vp;w=wb.+wp
    base_norm=velocity_norms(ub,vb,wb);perturb_norm=velocity_norms(up,vp,wp)
    base_native=native_velocity_rms(ub,vb,wb);perturb_native=native_velocity_rms(up,vp,wp)
    metrics=(;base_inplane_rms=base_norm.inplane,base_total_rms=base_norm.total,
        perturb_inplane_rms=perturb_norm.inplane,perturb_total_rms=perturb_norm.total,
        base_native_rms=base_native,perturb_native_rms=perturb_native,
        achieved_perturb_energy_ratio=base_norm.total>0 ? (perturb_norm.total/base_norm.total)^2 : nothing,
        achieved_native_perturb_energy_ratio=base_native>0 ? (perturb_native/base_native)^2 : nothing,
        max_native_divergence=maximum(abs,native_divergence(u,v,w,lx/nx,ly/ny,h/nz)),
        max_transverse_mean_perturbation=max(maximum(abs,mean(up;dims=2)),maximum(abs,mean(vp;dims=2)),maximum(abs,mean(wp;dims=2))),
        max_mean_profile_error=maximum(abs,vec(mean(u;dims=(1,2))).-background))
    return (;u,v,w,base=(ub,vb,wb),perturbation=(up,vp,wp),perturbation_shape,metrics)
end
