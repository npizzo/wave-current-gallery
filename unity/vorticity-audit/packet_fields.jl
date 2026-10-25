"""Analytic long-crested Gaussian packet, matching pinned moving_packet.jl.
Stokes u,w are averaged from their native face locations before use in curl.
The s derivatives are analytic cell-centre reference quantities, not native tendencies.
"""
function packet_fields(xf, zf, t, p; enabled=true)
    xc=(xf[1:end-1].+xf[2:end])./2
    zc=(zf[1:end-1].+zf[2:end])./2
    nx,nz=length(xc),length(zc)
    us=zeros(nx,nz); ws=similar(us); fill!(ws,0)
    s=copy(us); st=copy(us); sx=copy(us); sz=copy(us)
    enabled || return (;us,ws,s,s_t=st,s_x=sx,s_z=sz)
    k,A,sigma,cg,h=Float64.((p.k,p.Uˢ₀,p.σ₀,p.cᵍ,p.h))
    center=Float64(p.x₀)+cg*t
    Float64(p.periodic)==0 || error("This audit expects the bounded packet")
    g(x)=exp(-((x-center)/sigma)^2)
    gp(x)=-2(x-center)/sigma^2*g(x)
    gpp(x)=(4(x-center)^2/sigma^4-2/sigma^2)*g(x)
    gppp(x)=(12(x-center)/sigma^4-8(x-center)^3/sigma^6)*g(x)
    U(x,z)=A*g(x)*exp(2k*z)
    W(x,z)=-A/(2k)*gp(x)*(exp(2k*z)-exp(-2k*h))
    for j in 1:nz, i in 1:nx
        x,z=xc[i],zc[j]; e=exp(2k*z); f=e-exp(-2k*h)
        us[i,j]=(U(xf[i],z)+U(xf[i+1],z))/2
        ws[i,j]=(W(x,zf[j])+W(x,zf[j+1]))/2
        s[i,j]=-A/(2k)*gpp(x)*f-2k*A*g(x)*e
        sx[i,j]=-A/(2k)*gppp(x)*f-2k*A*gp(x)*e
        st[i,j]=-cg*sx[i,j]
        sz[i,j]=-A*gpp(x)*e-4k^2*A*g(x)*e
    end
    return (;us,ws,s,s_t=st,s_x=sx,s_z=sz)
end

function audit_zfaces(n,h)
    [Float64(h)*((1+(i/n-1)/1.5)*(1-exp(-8i/n))/(1-exp(-8))-1) for i in 0:n]
end
