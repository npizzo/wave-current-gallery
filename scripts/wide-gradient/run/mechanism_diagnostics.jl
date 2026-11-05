using Oceananigans.Advection: div_𝐯u, div_𝐯w
using Oceananigans.StokesDrifts: x_curl_Uˢ_cross_U, z_curl_Uˢ_cross_U
using Oceananigans.BoundaryConditions: fill_halo_regions!
using FFTW
@inline ma_u(i,j,k,g,a,u,v,w)=-div_𝐯u(i,j,k,g,a,(;u,v,w),u)
@inline ma_w(i,j,k,g,a,u,v,w)=-div_𝐯w(i,j,k,g,a,(;u,v,w),w)
@inline ms_u(i,j,k,g,sd,u,v,w,c)=x_curl_Uˢ_cross_U(i,j,k,g,sd,(;u,v,w),c.time)
@inline ms_w(i,j,k,g,sd,u,v,w,c)=z_curl_Uˢ_cross_U(i,j,k,g,sd,(;u,v,w),c.time)
function mechanism_init(m,out)
 g=m.grid;u,v,w=m.velocities
 u0=XFaceField(g);v0=YFaceField(g);w0=ZFaceField(g)
 function ops(a,u,v,w)
  (Field(KernelFunctionOperation{Face,Center,Center}(ma_u,g,a,u,v,w)),Field(KernelFunctionOperation{Center,Center,Face}(ma_w,g,a,u,v,w)))
 end
 full=ops(m.advection,u,v,w); avg=ops(m.advection,u0,v0,w0); centered=ops(Centered(order=2),u0,v0,w0)
 wave=(Field(KernelFunctionOperation{Face,Center,Center}(ms_u,g,m.stokes_drift,u,v,w,m.clock)),Field(KernelFunctionOperation{Center,Center,Face}(ms_w,g,m.stokes_drift,u,v,w,m.clock)))
 mkpath(joinpath(out,"transfer")); (;u0,v0,w0,full,avg,centered,wave)
end
mymean(f)=dropdims(mean(Array(interior(f));dims=2);dims=2)
function mechanism_sample!(d,m,out,t)
 for (dest,src) in zip((d.u0,d.v0,d.w0),m.velocities)
  a=Array(interior(src)); a.=mean(a;dims=2);set!(dest,a);fill_halo_regions!(dest)
 end
 arrays=Dict{String,Any}("u"=>mymean(m.velocities.u),"w"=>mymean(m.velocities.w))
 for key in (:full,:avg,:centered,:wave)
  pair=getproperty(d,key)
  for (component,f) in zip(("u","w"),pair)
   compute!(f);arrays[string(key,"_",component)]=mymean(f)
  end
 end
 # Native staggered fields, Float64; exact advection evaluated afresh, never cached RK tendencies.
 jldopen(joinpath(out,"transfer",@sprintf("t%012.4f.jld2",t)),"w") do f
  f["t"]=t
  for (k,a) in arrays; f[k]=a;end
 end
end
function wave_probe!(m,out,t)
 ub=mymean(m.velocities.u); nx,nz=size(ub)
 # Cosine projections on the native z centres, and Fourier x modes m=1..8.
 zeta=((1:nz).-0.5)./nz; a=fft(ub,1)./nx
 coeff=ComplexF64[sum(a[k+1,:].*cos.(n*π.*zeta))/nz for k in 1:min(8,nx÷2-1),n in 0:7]
 open(joinpath(out,"wave-probe.bin"),"a") do io;write(io,Float64(t));write(io,coeff);end
end
