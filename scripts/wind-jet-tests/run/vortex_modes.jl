using LinearAlgebra

"Finite-depth Rayleigh pencil, with actual UL and Eulerian q0' profiles."
function rayleigh_candidates(nz,h,k,S,alpha,uoffset,kx)
    dz=h/nz;z=[-h+j*dz for j in 1:nz-1]
    us=S.*exp.(2k.*z);UL=(1-alpha).*us.+uoffset;q0prime=-alpha*(2k)^2 .* us
    L=Matrix(SymTridiagonal(fill(-2/dz^2-kx^2,nz-1),fill(1/dz^2,nz-2)))
    A=Diagonal(UL)*L-Diagonal(q0prime)
    eig=eigen(A,L)
    urange=extrema([(1-alpha)*S*exp(-2k*h)+uoffset,(1-alpha)*S+uoffset])
    candidates=NamedTuple[]
    for j in eachindex(eig.values)
        c=complex(eig.values[j]);abs(imag(c))<=1e-9max(1,abs(real(c))) || continue
        cr=real(c);gap=max(urange[1]-cr,cr-urange[2])
        gap>1e-7max(S,1e-3) || continue
        phi=real.(eig.vectors[:,j]);maximum(abs,phi)>0 || continue
        phi./=maximum(abs,phi);phi[argmax(abs.(phi))]<0 && (phi .*= -1)
        nzvals=filter(x->abs(x)>1e-6,phi)
        nodes=count(i->sign(nzvals[i])!=sign(nzvals[i+1]),1:length(nzvals)-1)
        residual=norm(A*phi-cr*(L*phi))/max(norm(A*phi),abs(cr)*norm(L*phi),eps())
        push!(candidates,(;c=cr,phi=vcat(0.,phi,0.),gap,nodes,residual))
    end
    sort!(candidates;by=r->(r.nodes,-r.gap,r.c))
    isempty(candidates) && error("No isolated real vortex-wave mode outside UL range; change PROBE_MODE or ALPHA")
    return candidates
end

function finite_depth_mode(;nx,nz,lx,h,k,S,alpha,uoffset,horizontal_mode)
    kx=2pi*horizontal_mode/lx
    primary=first(rayleigh_candidates(nz,h,k,S,alpha,uoffset,kx))
    fine=rayleigh_candidates(2nz,h,k,S,alpha,uoffset,kx)
    same_nodes=filter(r->r.nodes==primary.nodes,fine)
    isempty(same_nodes) && error("Isolated mode cannot be matched under vertical refinement")
    refined=same_nodes[argmin(abs.([r.c-primary.c for r in same_nodes]))]
    dz=h/nz;phi=primary.phi;phiz=diff(phi)./dz
    kx_native=2sin(kx*(lx/nx)/2)/(lx/nx)
    # Native velocity energy for unit real cosine streamfunction amplitude.
    phic=(phi[1:end-1].+phi[2:end])./2
    norm2=(cos(kx*(lx/nx)/2)^2*sum(abs2,phiz)/nz+kx_native^2*sum(abs2,phic)/nz)/2
    return (;horizontal_mode,kx,kx_native,c=primary.c,omega=kx*primary.c,
        period=2pi/abs(kx*primary.c),phi_faces=phi,phi_z_centers=phiz,
        normalization_velocity_rms=sqrt(norm2),isolated_gap=primary.gap,
        refined_c=refined.c,relative_refinement_difference=abs(refined.c-primary.c)/max(abs(refined.c-uoffset),eps()),
        vertical_nodes=primary.nodes,eigen_residual=primary.residual,
        eigenproblem="(UL-c)*(phi_zz-kx^2*phi)-q0_prime*phi=0; phi=0 at both walls",
        discretization="second-order uniform-z Rayleigh eigenproblem; continuous horizontal kx; native discrete-curl initialization",
        sign_convention="psi real cosine; u=psi_z,w=-psi_x; Fourier exp(-i*kx*x), amplitude exp(-i*omega*t)")
end

function add_probe!(u,w,mode,lx,amplitude,sign)
    nx,ny,nz=size(u);dx=lx/nx
    psi_scale=sign*amplitude/mode.normalization_velocity_rms
    xf=collect(0:nx-1).*dx
    psi=psi_scale.*cos.(mode.kx.*xf).*transpose(mode.phi_faces)
    # Use the supplied native derivative, avoiding a second interpolation.
    u .+= reshape(psi_scale.*cos.(mode.kx.*xf).*transpose(mode.phi_z_centers),nx,1,nz)
    w .+= reshape(-(circshift(psi,(-1,0)).-psi)./dx,nx,1,nz+1)
    return psi_scale
end

function project_probe(ymeans,mode,lx,h)
    U,W=ymeans.U,ymeans.W
    nx,nz=size(U);xf=collect(0:nx-1).*lx/nx;xc=xf.+lx/(2nx)
    au=vec(transpose(exp.(-im*mode.kx.*xf))*U).*(2/nx)
    aw=vec(transpose(exp.(-im*mode.kx.*xc))*W).*(2/nx)
    bu=mode.phi_z_centers;bw=-im*mode.kx_native.*mode.phi_faces
    weights=ones(nz+1);weights[1]=.5;weights[end]=.5
    numerator=sum(conj.(bu).*au)+sum(weights.*conj.(bw).*aw)
    denominator=sum(abs2,bu)+sum(weights.*abs2.(bw))
    return numerator/denominator
end
