# This kernel would live in your solver.jl
using KernelAbstractions
using Atomix
using GWGrids

@inline function _get_deltaz(grid::PlanarRegularGrid, l::Int)
    if l == 1 # Top layer
        return grid.top - grid.botm[l]
    else
        return grid.botm[l-1] - grid.botm[l]
    end
end

@inline function _get_deltaz_unconf(grid::PlanarRegularGrid, l, h)
    # 1. Get the physical bottom and thickness (Geometry)
    bot = grid.botm[l]
    # Ideally, store top_elevs or compute cheaply:
    top = (l == 1) ? grid.top : grid.botm[l-1]

    # 2. Calculate Saturated Thickness (The Universal Formula)
    # If h > top (Confined)   --> min selects 'top' --> returns full_dz
    # If h < top (Unconfined) --> min selects 'h'   --> returns h - bot
    # If h < bot (Dry)        --> max selects 0.0
    
    # We essentially clamp the head 'h' to the cell geometry
    h_clamped = min(h, top)
    thickness = max(0.0, h_clamped - bot)

    return thickness
end

@inline function _get_conductance(K1, L1, K2, L2, Area)
    if K1 == 0.0 || K2 == 0.0
        return 0.0
    end
    return Area * (K1*K2) / (K2*L1 + K1*L2)
end

# --- 1. Internal Flow Kernel (The "Heart") ---
@kernel function compute_divergence_kernel!(du, @Const(u), p,
                                            @Const(grid::PlanarRegularGrid),
                                            _get_k,
                                            )
    l, r, c = @index(Global, NTuple)
    nlay, nrow, ncol = grid.nlay, grid.nrow, grid.ncol
    
    # Get cell dimensions
    delr_c = grid.delr[c]
    delc_r = grid.delc[r]
    deltaz_l = _get_deltaz(grid, l)
    
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
        L2 = _get_deltaz(grid, l-1) / 2.0
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
        L2 = _get_deltaz(grid, l+1) / 2.0
        K2z = _get_k(l+1, r, c, p, 3)
        cz_bot = _get_conductance(K1z, L1, K2z, L2, Area)
        
        idx_bot = idx + 1 # stride_l is 1
        h_bot = u[idx_bot]
        net_flux += cz_bot * (h_bot - h_curr)
    end

    # Store result in du (Rate of change of storage or Residual)
    du[idx] = net_flux
end


# --- 1. Internal Flow Kernel (The "Heart") ---
@kernel function compute_divergence_kernel_unconf!(du, @Const(u), p,
                                            @Const(grid::PlanarRegularGrid),
                                            _get_k,
                                            )
    l, r, c = @index(Global, NTuple)
    nlay, nrow, ncol = grid.nlay, grid.nrow, grid.ncol
    
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

    # Get cell dimensions
    delr_c = grid.delr[c]
    delc_r = grid.delc[r]
    deltaz_l = _get_deltaz_unconf(grid, l, h_curr)
    
    # Conductivity components for current cell
    K1x = _get_k(l, r, c, p, 1)
    K1y = _get_k(l, r, c, p, 2)
    K1z = _get_k(l, r, c, p, 3)

    # --- X-Direction (Columns) ---
    # Flux from Left (c-1 -> c)
    if c > 1
        idx_left = idx - stride_c
        h_left = u[idx_left]
        deltaz_c = _get_deltaz_unconf(grid, l, h_left)
        deltaz = max(deltaz_l, deltaz_c)
        Area = delc_r * deltaz
        L1 = delr_c / 2.0
        L2 = grid.delr[c-1] / 2.0
        K2x = _get_k(l, r, c-1, p, 1)
        cx_left = _get_conductance(K1x, L1, K2x, L2, Area)
        net_flux += cx_left * (h_left - h_curr)
    end
    
    # Flux from Right (c+1 -> c)
    if c < ncol
        idx_right = idx + stride_c
        h_right = u[idx_right]
        deltaz_r = _get_deltaz_unconf(grid, l, h_right)
        deltaz = max(deltaz_l, deltaz_r)
        Area = delc_r * deltaz
        L1 = delr_c / 2.0
        L2 = grid.delr[c+1] / 2.0
        K2x = _get_k(l, r, c+1, p, 1)
        cx_right = _get_conductance(K1x, L1, K2x, L2, Area)
        net_flux += cx_right * (h_right - h_curr)
    end

    # --- Y-Direction (Rows) ---
    # Flux from Top (r-1 -> r)
    if r > 1
        idx_up = idx - stride_r
        h_up = u[idx_up]
        deltaz_up = _get_deltaz_unconf(grid, l, h_up)
        deltaz = max(deltaz_l, deltaz_up)
        Area = delr_c * deltaz
        L1 = delc_r / 2.0
        L2 = grid.delc[r-1] / 2.0
        K2y = _get_k(l, r-1, c, p, 2)
        cy_top = _get_conductance(K1y, L1, K2y, L2, Area)
        net_flux += cy_top * (h_up - h_curr)
    end
    
    # Flux from Bottom (r+1 -> r)
    if r < nrow
        idx_down = idx + stride_r
        h_down = u[idx_down]
        deltaz_dwn = _get_deltaz_unconf(grid, l, h_down)
        deltaz = max(deltaz_l, deltaz_dwn)
        Area = delr_c * deltaz
        L1 = delc_r / 2.0
        L2 = grid.delc[r+1] / 2.0
        K2y = _get_k(l, r+1, c, p, 2)
        cy_bot = _get_conductance(K1y, L1, K2y, L2, Area)
        net_flux += cy_bot * (h_down - h_curr)
    end

    # --- Z-Direction (Layers) ---
    # Flux from Above (l-1 -> l)
    if l > 1
        Area = delr_c * delc_r
        L1 = deltaz_l / 2.0
        L2 = _get_deltaz(grid, l-1) / 2.0
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
        L2 = _get_deltaz(grid, l+1) / 2.0
        K2z = _get_k(l+1, r, c, p, 3)
        cz_bot = _get_conductance(K1z, L1, K2z, L2, Area)
        
        idx_bot = idx + 1 # stride_l is 1
        h_bot = u[idx_bot]
        net_flux += cz_bot * (h_bot - h_curr)
    end

    # Store result in du (Rate of change of storage or Residual)
    du[idx] = net_flux
end


# 1. Recursive Unrolling Helper
# This forces the compiler to generate specific code for each BC task, avoiding Unions.
@inline function _apply_bcs_recursively!(tasks::Tuple, du, u)
    # Call the first task in the list
    tasks[1](du, u) 
    # Recurse on the remaining tasks
    _apply_bcs_recursively!(Base.tail(tasks), du, u)
end

# Base case: When the tuple is empty, stop.
@inline function _apply_bcs_recursively!(::Tuple{}, du, u)
    return nothing
end

function make_rhs(grid::GWGrids.GWGrid, BCs::AbstractArray{BoundaryCondition}, _get_k::F) where F
    backend = get_backend(grid.delr)
    # 1. Sort BCs so CHD comes last (priority logic defined above)
    sort!(BCs)

    # check if the model is confined or unconfined:
    unconf = any(grid.li .> 0)
    if unconf
        ∇h! = compute_divergence_kernel_unconf!(backend)
    else
        ∇h! = compute_divergence_kernel!(backend)
    end
    
    # 2. Build a Tuple of Closures (Type Stable Barrier)
    # We define a helper to "capture" the specific types for each BC
    function build_task(bc)
        kernel! = apply_kernel!(backend)
        n_bc = length(bc.indices)
        # Return a specialized function for this specific BC type
        return (du, u) -> begin
            kernel!(du, u, bc; ndrange=n_bc)
            synchronize(backend) # <--- The AD-safe barrier
        end
    end

    # Convert Vector of BCs to a Tuple of Tasks
    # This forces the compiler to generate specialized code for each step
    bc_tasks = Tuple(map(build_task, BCs))

    #ndrange = grid.ndrange

    function rhs!(du, u, p, t)
        # 1. Internal Flow (Grid)
        ∇h!(du, u, p, grid, _get_k; ndrange=GWGrids.get_ndrange(grid))
        synchronize(backend) 

        # # 2. Boundary Conditions
        # for task! in bc_tasks
        #     # Update BC parameters from p if they are dynamic (e.g. time-varying)
        #     # _updateBC!(bc, p) 
        #     task!(du, u)
        # end
        _apply_bcs_recursively!(bc_tasks, du, u)
    end
    return rhs!
end

