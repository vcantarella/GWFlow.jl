
# --- 2. Boundary Condition Kernels ---
# Dispatch based on type
@kernel function apply_kernel!(du, @Const(u), @Const(fluxbc::FluxBC))
    i = @index(Global, Linear)
    idx = fluxbc.indices[i]
    q = fluxbc.flux[i]
    du[idx] += q
end

@kernel function apply_kernel!(du, @Const(u), @Const(ghb::GeneralHeadBC))
    i = @index(Global, Linear)
    idx = ghb.indices[i]
    h_ext = ghb.head[i]
    C_ghb = ghb.conductance[i]
    
    h_curr = u[idx]
    q_ghb = C_ghb * (h_ext - h_curr)
    du[idx] += q_ghb
end

@kernel function apply_kernel!(du, @Const(u), @Const(chb::ConstantHeadBC))
    i = @index(Global, Linear)
    idx = chb.indices[i]
    du[idx] = u[idx] - chb.head[i]
end

@kernel function apply_kernel!(du, @Const(u), @Const(drn::DrainBC))
    i = @index(Global, Linear)
    idx = drn.indices[i]
    h_ext = drn.stage[i]
    C_ghb = drn.conductance[i]
    
    h_curr = u[idx]
    if h_curr >= h_ext
        q_ghb = C_ghb * (h_ext - h_curr)
        du[idx] += q_ghb
    end
end

@kernel function apply_kernel!(du, @Const(u), @Const(riv::RiverBC))
    i = @index(Global, Linear)
    idx = riv.indices[i]
    h_ext = riv.stage[i]
    C_ghb = riv.conductance[i]
    bot = riv.bottom[i]
    
    h_curr = u[idx]
    if h_curr < bot
        q_ghb = C_ghb * (h_ext - bot)
    else
        q_ghb = C_ghb * (h_ext - h_curr)
    end
    du[idx] += q_ghb
end

@kernel function apply_kernel!(du, @Const(u), @Const(evt::EvapotranspirationBC))
    i = @index(Global, Linear)
    idx = evt.indices[i]
    surf = evt.surface[i]
    ext_depth = evt.ext_depth[i]
    evt_rate = evt.evt[i]
    
    h_curr = u[idx]
    surf_ext = surf - ext_depth
    if h_curr > surf
        q_ghb = evt_rate
    elseif  surf_ext ≤ h_curr ≤ surf 
        q_ghb = evt_rate * (h_curr - surf_ext)/ext_depth
    else # h < surf_ext
        q_ghb = 0.0
    end

    du[idx] -= q_ghb
end