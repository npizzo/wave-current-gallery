module PacketVorticityAuditMath

export snapshot_reductions

"""Three-node, quadratic-exact first-derivative weights, evaluated at x0."""
function first_weights(nodes, x0)
    a, b, c = nodes
    return ((2x0-b-c)/((a-b)*(a-c)),
            (2x0-a-c)/((b-a)*(b-c)),
            (2x0-a-b)/((c-a)*(c-b)))
end

function derivative_stencils(coordinates, step)
    n = length(coordinates)
    n >= 2step+1 || throw(ArgumentError("Need at least 2*stencil_step+1 nodes"))
    ids = Matrix{Int}(undef, 3, n)
    coefficients = Matrix{Float64}(undef, 3, n)
    for i in 1:n
        js = i <= step ? (i, i+step, i+2step) :
             i > n-step ? (i-2step, i-step, i) : (i-step, i, i+step)
        weights = first_weights(ntuple(j -> coordinates[js[j]], 3), coordinates[i])
        for j in 1:3
            ids[j,i] = js[j]
            coefficients[j,i] = weights[j]
        end
    end
    return (; ids, coefficients)
end

@inline function derivative_x(a, i, j, k, stencil)
    @inbounds return stencil.coefficients[1,i]*a[stencil.ids[1,i],j,k] +
                    stencil.coefficients[2,i]*a[stencil.ids[2,i],j,k] +
                    stencil.coefficients[3,i]*a[stencil.ids[3,i],j,k]
end

@inline function derivative_z(a, i, j, k, stencil)
    @inbounds return stencil.coefficients[1,k]*a[i,j,stencil.ids[1,k]] +
                    stencil.coefficients[2,k]*a[i,j,stencil.ids[2,k]] +
                    stencil.coefficients[3,k]*a[i,j,stencil.ids[3,k]]
end

@inline function derivative_x2(a, i, k, stencil)
    @inbounds return stencil.coefficients[1,i]*a[stencil.ids[1,i],k] +
                    stencil.coefficients[2,i]*a[stencil.ids[2,i],k] +
                    stencil.coefficients[3,i]*a[stencil.ids[3,i],k]
end

@inline function derivative_z2(a, i, k, stencil)
    @inbounds return stencil.coefficients[1,k]*a[i,stencil.ids[1,k]] +
                    stencil.coefficients[2,k]*a[i,stencil.ids[2,k]] +
                    stencil.coefficients[3,k]*a[i,stencil.ids[3,k]]
end

@inline periodic_forward(j, step, n) = j+step > n ? j+step-n : j+step
@inline periodic_backward(j, step, n) = j-step < 1 ? j-step+n : j-step
@inline derivative_y(a, i, j, k, step, dy, ny) =
    @inbounds (a[i,periodic_forward(j,step,ny),k] - a[i,periodic_backward(j,step,ny),k]) / (2step*dy)

function coefficient_field(a, nx, nz, name)
    a === nothing && return zeros(Float64, nx, nz)
    size(a) == (nx,nz) || throw(DimensionMismatch("$name must have shape (nx,nz)"))
    all(isfinite, a) || throw(ArgumentError("Nonfinite $name"))
    return a
end

"""
    snapshot_reductions(u, v, w, dx, dy, zfaces;
                        us=nothing, ws=nothing,
                        s=nothing, s_t=nothing, s_x=nothing, s_z=nothing,
                        stencil_step=1)

Read-only diagnostic for full Lagrangian velocities on an x,z-bounded,
y-periodic C grid. Input shapes are (nx+1,ny,nz), (nx,ny,nz),
(nx,ny,nz+1). The optional us/ws are prescribed Stokes components at cell
centres, shape (nx,nz); they must be independent of y. No null run, horizontal
mean, or ensemble mean is removed. q is the y curl of the Eulerian velocity,
q = d_z(u-us) - d_x(w-ws). Advecting velocities remain Lagrangian.
For an interpolation-consistent Eulerian subtraction, form us/ws by averaging
their values at the same adjacent native faces as u/w. Evaluating the analytic
Stokes functions directly at geometric centres is a different interpolation.

All returned dictionaries contain x,z matrices: `means` and `mean_squares`
average over y, and `maxabs` takes the maximum absolute value over y. Region
averaging must subsequently use physical dz weights. `mean_squares["R3D"]`
is the squared NET residual, not the sum of squared stretching terms.

Native ux,vy,wz use adjacent staggered faces and never change with
stencil_step. All other derivatives use quadratic-exact three-point x/z
stencils and centred periodic y differences with spacing step*dy. Step=2
is a diagnostic stencil sensitivity check, not a resolution study. At x,z
edges the stencils are one-sided. Users comparing q derivatives should omit
at least 2*step cells at each x,z edge if edge accuracy is not being studied.

Optional s/s_t/s_x/s_z refer to s = -d_z(us)+d_x(ws), the y curl contribution
needed for eta=curl_y(u_L)=q-s. `s_material`=s_t+u*s_x+w*s_z and
`eta_rhs`=R3D-s_material; `s_vy` uses centred v_y. eta has the opposite
sign to the convention zeta=d_x(w)-d_z(u). If absent, the source coefficients are zero, and
`source_supplied` is false. These terms exclude numerical/viscous forces.

Peak work storage is four Float64 volumes (centred u,v,w,q), plus x,z
reductions; no derivative volumes or native input copies are retained.
"""
function snapshot_reductions(u, v, w, dx, dy, zfaces;
                             us=nothing, ws=nothing,
                             s=nothing, s_t=nothing, s_x=nothing, s_z=nothing,
                             stencil_step::Integer=1)
    stencil_step in (1,2) || throw(ArgumentError("stencil_step must be 1 or 2"))
    ndims(v) == 3 || throw(DimensionMismatch("v must be a 3-D array"))
    nx,ny,nz = size(v)
    size(u) == (nx+1,ny,nz) || throw(DimensionMismatch("u must be on bounded x faces"))
    size(w) == (nx,ny,nz+1) || throw(DimensionMismatch("w must be on bounded z faces"))
    length(zfaces) == nz+1 || throw(DimensionMismatch("zfaces must have nz+1 entries"))
    ny > 2stencil_step || throw(ArgumentError("Need ny>2*stencil_step periodic cells"))
    isfinite(dx) && isfinite(dy) && dx>0 && dy>0 || throw(ArgumentError("Positive finite dx,dy required"))
    all(isfinite,zfaces) || throw(ArgumentError("Nonfinite zfaces"))
    dz = diff(Float64.(zfaces))
    all(>(0),dz) || throw(ArgumentError("zfaces must be strictly increasing"))
    z = (Float64.(zfaces[1:end-1]) .+ zfaces[2:end]) ./ 2
    x = ((collect(1:nx) .- 0.5) .* Float64(dx))
    xs = derivative_stencils(x,stencil_step)
    zs = derivative_stencils(z,stencil_step)
    source_supplied = any(a -> a !== nothing, (s_t,s_x,s_z))
    us = coefficient_field(us,nx,nz,"us")
    ws = coefficient_field(ws,nx,nz,"ws")
    sfield = coefficient_field(s,nx,nz,"s")
    st = coefficient_field(s_t,nx,nz,"s_t")
    sx = coefficient_field(s_x,nx,nz,"s_x")
    sz = coefficient_field(s_z,nx,nz,"s_z")

    uc = Array{Float64}(undef,nx,ny,nz)
    vc = similar(uc); wc = similar(uc); q = similar(uc)
    max_native_velocity = 0.0
    for a in (u,v,w)
        all(isfinite,a) || throw(ArgumentError("Nonfinite input velocity"))
        max_native_velocity = max(max_native_velocity, maximum(abs,a))
    end
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        jp = periodic_forward(j,1,ny)
        uc[i,j,k] = (Float64(u[i,j,k])+Float64(u[i+1,j,k]))/2
        vc[i,j,k] = (Float64(v[i,j,k])+Float64(v[i,jp,k]))/2
        wc[i,j,k] = (Float64(w[i,j,k])+Float64(w[i,j,k+1]))/2
    end
    curl_stokes_y = zeros(Float64,nx,nz)
    @inbounds for k in 1:nz, i in 1:nx
        curl_stokes_y[i,k] = derivative_z2(us,i,k,zs)-derivative_x2(ws,i,k,xs)
    end
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        q[i,j,k] = derivative_z(uc,i,j,k,zs)-derivative_x(wc,i,j,k,xs)-curl_stokes_y[i,k]
    end

    names = ("native_ux","native_vy","native_wz","native_divergence","native_xz_divergence",
             "u","v","w","u_x","v_y","w_z","xz_divergence","u_y","w_y","v_x","v_z",
             "q","qx","qy","qz","u_qx","w_qz","adv_xz","v_qy",
             "omegaEx_vx","q_vy","omegaEz_vz","stretching","R3D",
             "s_vy","s_t","u_sx","w_sz","s_material","eta_rhs")
    means_arrays = [zeros(Float64,nx,nz) for _ in names]
    square_arrays = [zeros(Float64,nx,nz) for _ in names]
    max_arrays = [zeros(Float64,nx,nz) for _ in names]
    identity_error_by_z = zeros(Float64,nz)
    # Each thread owns complete x,z reduction columns, with fixed y summation
    # order. No atomics, cross-thread reductions, or 3-D derivative arrays.
    Threads.@threads for k in 1:nz
      identity_error = 0.0
      @inbounds for j in 1:ny, i in 1:nx
        jp = periodic_forward(j,1,ny)
        ux_n = (Float64(u[i+1,j,k])-Float64(u[i,j,k]))/dx
        vy_n = (Float64(v[i,jp,k])-Float64(v[i,j,k]))/dy
        wz_n = (Float64(w[i,j,k+1])-Float64(w[i,j,k]))/dz[k]
        divergence = ux_n+vy_n+wz_n
        ux = derivative_x(uc,i,j,k,xs)
        vy = derivative_y(vc,i,j,k,stencil_step,dy,ny)
        wz = derivative_z(wc,i,j,k,zs)
        uy = derivative_y(uc,i,j,k,stencil_step,dy,ny)
        wy = derivative_y(wc,i,j,k,stencil_step,dy,ny)
        vx = derivative_x(vc,i,j,k,xs)
        vz = derivative_z(vc,i,j,k,zs)
        qx = derivative_x(q,i,j,k,xs)
        qy = derivative_y(q,i,j,k,stencil_step,dy,ny)
        qz = derivative_z(q,i,j,k,zs)
        u0,v0,w0,q0 = uc[i,j,k],vc[i,j,k],wc[i,j,k],q[i,j,k]
        ax,az,ay = u0*qx,w0*qz,v0*qy
        tilt_x,stretch_y,tilt_z = (wy-vz)*vx,q0*vy,(vx-uy)*vz
        stretch = tilt_x+stretch_y+tilt_z
        residual = stretch-ay
        identity_error = max(identity_error,abs(residual-(wy*vx-uy*vz+q0*vy-ay)))
        st0,source_x,source_z = st[i,k],u0*sx[i,k],w0*sz[i,k]
        source = st0+source_x+source_z
        values = (ux_n,vy_n,wz_n,divergence,ux_n+wz_n,
                  u0,v0,w0,ux,vy,wz,ux+wz,uy,wy,vx,vz,
                  q0,qx,qy,qz,ax,az,ax+az,ay,
                  tilt_x,stretch_y,tilt_z,stretch,residual,
                  sfield[i,k]*vy,st0,source_x,source_z,source,residual-source)
        for n in eachindex(names)
            value = values[n]
            isfinite(value) || throw(ArgumentError("Nonfinite diagnostic $(names[n])"))
            means_arrays[n][i,k] += value
            square_arrays[n][i,k] += value*value
            max_arrays[n][i,k] = max(max_arrays[n][i,k],abs(value))
        end
      end
      identity_error_by_z[k] = identity_error
    end
    for a in means_arrays
        a ./= ny
    end
    for a in square_arrays
        a ./= ny
    end
    means = Dict(name=>a for (name,a) in zip(names,means_arrays))
    mean_squares = Dict(name=>a for (name,a) in zip(names,square_arrays))
    maxabs = Dict(name=>a for (name,a) in zip(names,max_arrays))
    max_native_divergence = maximum(maxabs["native_divergence"])
    scaled_divergence = max_native_velocity==0 ? max_native_divergence :
        max_native_divergence*min(Float64(dx),Float64(dy),minimum(dz))/max_native_velocity
    qc = (; max_native_divergence, scaled_native_divergence_max=scaled_divergence,
            R3D_identity_error=maximum(identity_error_by_z),
            max_native_velocity, stencil_step, source_supplied,
            nx,ny,nz, recommended_edge_exclusion=2stencil_step)
    return (;x,z,dz,means,mean_squares,maxabs,qc)
end

end # module
