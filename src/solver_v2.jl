# This kernel would live in your solver.jl
using KernelAbstractions
using Atomix
using GWGrids

@inline function _get_deltaz_v2(grid::PlanarRegularGrid, l::Int)
    if l == 1 # Top layer
        return grid.top - grid.botm[l]
    else
        return grid.botm[l-1] - grid.botm[l]
    end
end


@inline function _get_conductance(K1, L1, K2, L2, Area)
    if K1 == 0.0 || K2 == 0.0
        return 0.0
    end
    return Area * (K1*K2) / (K2*L1 + K1*L2)
end

# --- 1. Internal Flow Kernel (The "Heart") ---
@kernel function compute_divergence_kernel!(du, @Const(u), p,
                                            @Const(grid),
                                            _get_k::Function,
                                            )
    l, r, c = @index(Global, NTuple)
    nlay, nrow, ncol = grid.nlay, grid.nrow, grid.ncol
    
    # Get cell dimensions
    delr_c = grid.delr[c]
    delc_r = grid.delc[r]
    deltaz_l = _get_deltaz_v2(grid, l)
    
    # Linear index for the 1D vectors du and u
    # Layout: (nlay, nrow, ncol) -> Column Major
    # Fastest: l, then r, then c
    idx = GWGrids.get_linear_index(grid, l, r,c)
    
    # Strides
    # stride_l = 1
    stride_r = nlay
    stride_c = nlay * nrow

    # Initialize divergence (net flux IN)
    net_flux = zero(eltype(du))
    
    # Current head
    h_curr = u[idx]
    
    # Conductivity components for current cell
    K1x = _get_k(l, r, c, p, 1)
    K1y = _get_k(l, r, c, p, 2)
    K1z = _get_k(l, r, c, p, 3)

    # --- X-Direction (Columns) ---
    # Flux from Left (c-1 -> c)
    if c > 1
        Area = delc_r * deltaz_l
        L1 = delr_c / 2.0
        L2 = grid.delr[c-1] / 2.0
        K2x = _get_k(l, r, c-1, p, 1)
        cx_left = _get_conductance(K1x, L1, K2x, L2, Area)
        
        idx_left = idx - stride_c
        h_left = u[idx_left]
        net_flux += cx_left * (h_left - h_curr)
    end
    
    # Flux from Right (c+1 -> c)
    if c < ncol
        Area = delc_r * deltaz_l
        L1 = delr_c / 2.0
        L2 = grid.delr[c+1] / 2.0
        K2x = _get_k(l, r, c+1, p, 1)
        cx_right = _get_conductance(K1x, L1, K2x, L2, Area)
        
        idx_right = idx + stride_c
        h_right = u[idx_right]
        net_flux += cx_right * (h_right - h_curr)
    end

    # --- Y-Direction (Rows) ---
    # Flux from Top (r-1 -> r)
    if r > 1
        Area = delr_c * deltaz_l
        L1 = delc_r / 2.0
        L2 = grid.delc[r-1] / 2.0
        K2y = _get_k(l, r-1, c, p, 2)
        cy_top = _get_conductance(K1y, L1, K2y, L2, Area)
        
        idx_up = idx - stride_r
        h_up = u[idx_up]
        net_flux += cy_top * (h_up - h_curr)
    end
    
    # Flux from Bottom (r+1 -> r)
    if r < nrow
        Area = delr_c * deltaz_l
        L1 = delc_r / 2.0
        L2 = grid.delc[r+1] / 2.0
        K2y = _get_k(l, r+1, c, p, 2)
        cy_bot = _get_conductance(K1y, L1, K2y, L2, Area)
        
        idx_down = idx + stride_r
        h_down = u[idx_down]
        net_flux += cy_bot * (h_down - h_curr)
    end

    # --- Z-Direction (Layers) ---
    # Flux from Above (l-1 -> l)
    if l > 1
        Area = delr_c * delc_r
        L1 = deltaz_l / 2.0
        L2 = _get_deltaz_v2(grid, l-1) / 2.0
        K2z = _get_k(l-1, r, c, p, 3)
        cz_top = _get_conductance(K1z, L1, K2z, L2, Area)
        
        idx_top = idx - 1
        h_top = u[idx_top]
        net_flux += cz_top * (h_top - h_curr)
    end
    
    # Flux from Below (l+1 -> l)
    if l < nlay
        Area = delr_c * delc_r
        L1 = deltaz_l / 2.0
        L2 = _get_deltaz_v2(grid, l+1) / 2.0
        K2z = _get_k(l+1, r, c, p, 3)
        cz_bot = _get_conductance(K1z, L1, K2z, L2, Area)
        
        idx_bot = idx + 1 # stride_l is 1
        h_bot = u[idx_bot]
        net_flux += cz_bot * (h_bot - h_curr)
    end

    # Store result in du (Rate of change of storage or Residual)
    du[idx] = net_flux
end


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
    du[idx] = 0.0 
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


function make_rhs(grid, BCs, _get_k::Function)
    backend = get_backend(grid.delr)
    # 1. Sort BCs so CHD comes last (priority logic defined above)
    sort!(BCs)
    
    ∇h! = compute_divergence_kernel!(backend)
    
    # 2. Pre-compile BC tasks
    # We store a tuple of (task, bc) so we can update the BC object
    # before running the task.
    bc_ops = map(BCs) do bc
        kernel! = apply_kernel!(backend)
        n_bc = length(bc.indices)
        
        task! = (du, u) -> begin
            kernel!(du, u, bc; ndrange=n_bc)
            synchronize(backend)
        end
        
        return task!
    end

    function rhs!(du, u, p, t)
        # 1. Internal Flow (Grid)
        ∇h!(du, u, p, grid, _get_k; ndrange=(grid.nlay, grid.nrow, grid.ncol))
        synchronize(backend) 

        # 2. Boundary Conditions
        for task! in bc_ops
            # Update BC parameters from p if they are dynamic (e.g. time-varying)
            # _updateBC!(bc, p) 
            task!(du, u)
        end
    end
    return rhs!
end

