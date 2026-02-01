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

@inline function _get_deltaz_convert(grid::PlanarRegularGrid{T}, l::Int,
    h_this::T, h_next::T) where T
    if grid.li[l] == 0 # convertible/confined check
        return _get_deltaz(grid, l)
    else 
        # Calculate saturated thickness based on the maximum head of the two connected cells (Upstream Weighting)
        hmax = max(h_this, h_next)
        
        maxdz = _get_deltaz(grid, l)
        # Saturated thickness: max(0, head - bottom) clamped to cell thickness
        return min(maxdz, max(0.0, hmax - grid.botm[l]))
    end
end


@inline function _get_conductance(K1, L1, K2, L2, Area)
    if K1 == 0.0 || K2 == 0.0
        return 0.0
    end
    return Area * (K1*K2) / (K2*L1 + K1*L2)
end

struct SolverCachePlanar{T, A <: AbstractArray{T, 3}, V <: AbstractVector{T}}
    Cx::A
    Cy::A
    Cz::A
    A_diag::V
    b::V
end

function get_solver_cache(model::FlowModel{<:PlanarRegularGrid}, backend)
    # Check if cache exists and is valid
    if model.cache isa SolverCachePlanar
        cache = model.cache
        # Optional: check dimensions if grid might change (unlikely for now)
        return cache
    end

    # Allocate new cache
    grid = model.grid
    nlay, nrow, ncol = grid.nlay, grid.nrow, grid.ncol
    n_nodes = nlay * nrow * ncol
    
    # Use properties array as prototype for type/device
    proto = model.properties.k_horiz
    T = eltype(proto)

    Cx = similar(proto, T, (nlay, nrow, ncol))
    Cy = similar(proto, T, (nlay, nrow, ncol))
    Cz = similar(proto, T, (nlay, nrow, ncol))
    A = similar(proto, T, n_nodes)
    b = similar(proto, T, n_nodes)

    cache = SolverCachePlanar(Cx, Cy, Cz, A, b)
    
    # Store in model if possible (assuming mutable FlowModel with Any typed cache)
    if isdefined(model, :cache)
        model.cache = cache
    end
    
    return cache
end

@kernel function compute_conductances_kernel!(Cx, Cy, Cz, 
    @Const(grid), @Const(k_row), @Const(k_col),@Const(k_vert))
    l, r, c = @index(Global, NTuple)
    nlay, nrow, ncol = grid.nlay, grid.nrow, grid.ncol

    # Get cell dimensions
    delr_c = grid.delr[c]
    delc_r = grid.delc[r]
    deltaz_l = _get_deltaz(grid, l)

    # --- X-direction Conductance ---
    if c < ncol
        Area = delc_r * deltaz_l
        L1 = delr_c / 2.0
        L2 = grid.delr[c+1] / 2.0
        K1 = k_row[l, r, c]
        K2 = k_row[l, r, c+1]
        Cx[l, r, c] = _get_conductance(K1, L1, K2, L2, Area)
    end
    
    # --- Y-direction Conductance ---
    if r < nrow
        Area = delr_c * deltaz_l
        L1 = delc_r / 2.0
        L2 = grid.delc[r+1] / 2.0
        K1 = k_col[l, r, c]
        K2 = k_col[l, r+1, c]
        Cy[l, r, c] = _get_conductance(K1, L1, K2, L2, Area)
    end
    
    # --- Z-direction Conductance ---
    if l < nlay
        Area = delr_c * delc_r
        L1 = deltaz_l / 2.0
        L2 = _get_deltaz(grid, l+1) / 2.0
        K1 = k_vert[l, r, c]
        K2 = k_vert[l+1, r, c]
        Cz[l, r, c] = _get_conductance(K1, L1, K2, L2, Area)
    end
end

@kernel function compute_conductances_kernel_unconf!(Cx, Cy, Cz, 
    @Const(grid), @Const(k_row), @Const(k_col), @Const(h))
    l, r, c = @index(Global, NTuple)
    nlay, nrow, ncol = grid.nlay, grid.nrow, grid.ncol

    # Get cell dimensions
    delr_c = grid.delr[c]
    delc_r = grid.delc[r]

    # get the current head:
    h_this = h[l, r, c]

    # --- X-direction Conductance (Between Columns) ---
    if c < ncol
        #X-direction
        h_next = h[l, r, c+1]
        delta_z = _get_deltaz_convert(grid, l, h_this, h_next)
        Area = delc_r * delta_z
        L1 = delr_c / 2.0
        L2 = grid.delr[c+1] / 2.0
        K1 = k_row[l, r, c]
        K2 = k_row[l, r, c+1]
        Cx[l, r, c] = _get_conductance(K1, L1, K2, L2, Area)
    end
    
    # --- Y-direction Conductance (Between Rows) ---
    if r < nrow
        #Y-direction
        h_next = h[l, r+1, c]
        delta_z = _get_deltaz_convert(grid, l, h_this, h_next)
        Area = delr_c * delta_z
        L1 = delc_r / 2.0
        L2 = grid.delc[r+1] / 2.0
        K1 = k_col[l, r, c]
        K2 = k_col[l, r+1, c]
        Cy[l, r, c] = _get_conductance(K1, L1, K2, L2, Area)
    end
    
    # --- Z-direction Conductance (Between Layers) ---
    if l < nlay
        #For vertical, the unsaturated flow depends on the heads too.
        # if heads are below the bottom, it becomes a vertical pass through
        deltaz_l = _get_deltaz(grid, l)
        Area = delr_c * delc_r
        L1 = deltaz_l / 2.0
        L2 = _get_deltaz(grid, l+1) / 2.0
        K1 = k_vert[l, r, c]
        K2 = k_vert[l+1, r, c]
        Cz[l, r, c] = _get_conductance(K1, L1, K2, L2, Area)
    end
end


# --- 1. Internal Flow Kernel (The "Heart") ---
@kernel function compute_divergence_kernel!(du, @Const(u), @Const(Cx), @Const(Cy), @Const(Cz), 
                                            @Const(grid))
    l, r, c = @index(Global, NTuple)
    nlay, nrow, ncol = grid.nlay, grid.nrow, grid.ncol
    
    # Linear index for the 1D vectors du and u
    # (Assuming grid maps (l,r,c) -> i linearly)
    idx = (l-1)*nrow*ncol + (r-1)*ncol + c

    # Initialize divergence (net flux IN)
    net_flux = zero(eltype(du))
    
    # Current head
    h_curr = u[idx]

    # --- X-Direction (Columns) ---
    # Flux from Left (c-1 -> c)
    if c > 1
        idx_left = idx - 1
        h_left = u[idx_left]
        cond = Cx[l, r, c-1] # Cx stored at i-1/2 face
        net_flux += cond * (h_left - h_curr)
    end
    # Flux to Right (c -> c+1)
    if c < ncol
        idx_right = idx + 1
        h_right = u[idx_right]
        cond = Cx[l, r, c] # Cx stored at i+1/2 face
        net_flux += cond * (h_right - h_curr)
    end

    # --- Y-Direction (Rows) ---
    # Flux from Top (r-1 -> r)
    if r > 1
        idx_up = idx - ncol
        h_up = u[idx_up]
        cond = Cy[l, r-1, c]
        net_flux += cond * (h_up - h_curr)
    end
    # Flux to Bottom (r -> r+1)
    if r < nrow
        idx_down = idx + ncol
        h_down = u[idx_down]
        cond = Cy[l, r, c]
        net_flux += cond * (h_down - h_curr)
    end

    # --- Z-Direction (Layers) ---
    # Flux from Above (l-1 -> l)
    if l > 1
        idx_top = idx - (nrow * ncol)
        h_top = u[idx_top]
        cond = Cz[l-1, r, c]
        net_flux += cond * (h_top - h_curr)
    end
    # Flux to Below (l -> l+1)
    if l < nlay
        idx_bot = idx + (nrow * ncol)
        h_bot = u[idx_bot]
        cond = Cz[l, r, c]
        net_flux += cond * (h_bot - h_curr)
    end

    # Store result in du (Rate of change of storage or Residual)
    # Note: For steady state, we want this to be 0.
    # For transient: Sy * dh/dt = net_flux + Q_ext
    # So: du = (net_flux + Q_ext) / Sy
    # Here we just store net_flux; division by Sy happens later or in separate kernel
    du[idx] = net_flux
end

# --- 2. Boundary Condition Kernels ---

@kernel function apply_flux_bc_kernel!(du, @Const(indices), @Const(fluxes))
    i = @index(Global, Linear)
    idx = indices[i]
    q = fluxes[i]
    # Add external flux source to the divergence
    Atomix.@atomic du[idx] += q
end

@kernel function apply_ghb_bc_kernel!(du, @Const(u), @Const(indices), @Const(heads), @Const(conds))
    i = @index(Global, Linear)
    idx = indices[i]
    h_ext = heads[i]
    C_ghb = conds[i]
    
    # Flux = C * (h_ext - h_cell)
    # This adds to the net_flux accumulator in du
    h_curr = u[idx]
    q_ghb = C_ghb * (h_ext - h_curr)
    
    Atomix.@atomic du[idx] += q_ghb
end

@kernel function apply_chb_kernel!(du, @Const(indices))
    i = @index(Global, Linear)
    idx = indices[i]
    # Force the rate of change to zero
    # implying dh/dt = 0, so h stays constant
    du[idx] = 0.0
end

# --- Helpers to Extract BCs to Backend Arrays ---

# Simplified helper to create backend array from CPU vector
function _to_backend(backend, data::Vector{T}) where T
    arr = KernelAbstractions.allocate(backend, T, length(data))
    copyto!(arr, data)
    return arr
end

function _extract_flux_bcs(model, backend)
    indices = Int[]
    values = Float64[]
    
    for bc in values(model.conditions)
        if bc isa FluxBC
            # Handle both scalar and vector inputs
            idx_list = (bc.indices isa Vector) ? bc.indices : [bc.indices]
            flux_list = (bc.flux isa Vector) ? bc.flux : fill(bc.flux, length(idx_list))
            
            append!(indices, idx_list)
            append!(values, flux_list)
        end
    end
    
    if isempty(indices)
        return nothing
    end
    
    return (_to_backend(backend, indices), _to_backend(backend, values))
end

function _extract_ghb_bcs(model, backend)
    indices = Int[]
    heads = Float64[]
    conds = Float64[]
    
    for bc in values(model.conditions)
        if bc isa GeneralHeadBC
            idx_list = (bc.indices isa Vector) ? bc.indices : [bc.indices]
            h_list = (bc.head isa Vector) ? bc.head : fill(bc.head, length(idx_list))
            c_list = (bc.conductance isa Vector) ? bc.conductance : fill(bc.conductance, length(idx_list))
            
            append!(indices, idx_list)
            append!(heads, h_list)
            append!(conds, c_list)
        end
    end
    
    if isempty(indices)
        return nothing
    end
    
    return (_to_backend(backend, indices), _to_backend(backend, heads), _to_backend(backend, conds))
end

function _extract_chb_indices(model, backend)
    indices = Int[]
    
    for bc in values(model.conditions)
        if bc isa ConstantHeadBC
            idx_list = (bc.indices isa Vector) ? bc.indices : [bc.indices]
            append!(indices, idx_list)
        end
    end
    
    if isempty(indices)
        return nothing
    end
    
    return _to_backend(backend, unique(indices))
end

# --- 3. The Function Constructor ---

function build_groundwater_function(model, backend)
    # 1. Pre-calculate Conductances (Cx, Cy, Cz) once if they are constant (Confined)
    #    If unconfined, these need to be recalculated INSIDE f! (Non-linear).
    #    Let's assume we pre-calc them for now for simplicity.
    cache = get_solver_cache(model, backend)

    # Run the conductance kernel you wrote earlier to populate cache.Cx, Cy, Cz
    k_cond = compute_conductances_kernel!(backend)
    k_cond(cache.Cx, cache.Cy, cache.Cz, model.grid, 
           model.properties.k_horiz, model.properties.k_col, model.properties.k_vert;
           ndrange=(model.grid.nlay, model.grid.nrow, model.grid.ncol))
    synchronize(backend)

    # 2. Extract BC arrays to GPU
    flux_bcs = _extract_flux_bcs(model, backend) 
    ghb_bcs  = _extract_ghb_bcs(model, backend)
    chb_indices = _extract_chb_indices(model, backend) 

    # 3. Create the ODE function
    function f!(du, u, p, t)
        fill!(du, 0.0)
        
        # 1. Calculate Physics (Fluxes)
        compute_divergence_kernel!(backend)(
            du, u, cache.Cx, cache.Cy, cache.Cz, model.grid;
            ndrange=(model.grid.nlay, model.grid.nrow, model.grid.ncol)
        )
        
        # 2. Apply Flux & GHB BCs
        if flux_bcs !== nothing
            apply_flux_bc_kernel!(backend)(du, flux_bcs[1], flux_bcs[2]; ndrange=length(flux_bcs[1]))
        end
        
        if ghb_bcs !== nothing
            apply_ghb_bc_kernel!(backend)(du, u, ghb_bcs[1], ghb_bcs[2], ghb_bcs[3]; ndrange=length(ghb_bcs[1]))
        end
        
        # 3. Apply Constant Head (The Overwrite)
        if chb_indices !== nothing
            apply_chb_kernel!(backend)(du, chb_indices; ndrange=length(chb_indices))
        end
        
        # 4. Handle Storage (Sy) - Optional, depends on solver (IDA vs ODESolver)
        # du .= du ./ Sy 
    end
    
    return f!, chb_indices
end

function solve_steady_state(model, backend)
    # 1. Build the function f!(du, u, p)
    # Note: NonlinearSolve expects f!(du, u, p), not (du, u, p, t)
    # We can wrap your existing function easily.
    f_ode!, chb_indices = build_groundwater_function(model, backend)
    f!(du, u, p) = f_ode!(du, u, p, 0.0) 

    # 2. Define Initial Guess
    # Important: Set the Constant Head values in u0 correctly!
    # Since f! returns 0.0 for CH cells, the solver won't change them.
    u0 = deepcopy(model.initial_heads) # Move to GPU first

    # 3. Define the Problem
    prob = NonlinearProblem(f!, u0, model.parameters)

    # 4. Select the Linear Solver Strategy
    # We use KrylovJL_GMRES because we don't have a matrix.
    # It estimates J*v using finite differences on your f! function.
    ls = LinearSolve.KrylovJL_GMRES()

    # 5. Solve
    # NewtonRaphson will call GMRES to find the update direction
    sol = solve(prob, 
                NewtonRaphson(linsolve = ls), 
                abstol=1e-5, reltol=1e-5)
    
    return sol.u
end

# @kernel function build_diag_rhs_kernel!(
#     A_diag, b,
#     @Const(Cx), @Const(Cy), @Const(Cz),
#     @Const(grid)
# )
#     l, r, c = @index(Global, NTuple) # 3D thread index
#     idx = _to_linear_index(grid, l, r, c) # Assuming this is GPU-safe

#     T = eltype(Cx)

#     nlay, nrow, ncol = grid.nlay, grid.nrow, grid.ncol

#     diag_val = zero(T)

#     if c > 1;   diag_val += Cx[l, r, c-1]; end
#     if c < ncol; diag_val += Cx[l, r, c];   end
#     if r > 1;   diag_val += Cy[l, r-1, c]; end
#     if r < nrow; diag_val += Cy[l, r, c];   end
#     if l > 1;   diag_val += Cz[l-1, r, c]; end
#     if l < nlay; diag_val += Cz[l, r, c];   end
    
#     # --- FIX FOR SINGULAR MATRIX ---
#     if diag_val == zero(T)
#         # This is an INACTIVE node (all K=0)
#         # Set diagonal to 1.0 to make matrix non-singular.
#         A_diag[idx] = 1.0
#         b[idx] = zero(T)
#     else
#         # This is an ACTIVE node
#         A_diag[idx] = -diag_val
#         b[idx] = zero(T)
#     end
# end

# @kernel function apply_ghb_kernel!(A_diag, b,
#      @Const(indices), @Const(heads), @Const(conds))
#     i = @index(Global, Linear) # 1D thread index
    
#     idx = indices[i]
#     h = (heads isa AbstractVector) ? heads[i] : heads
#     C = (conds isa AbstractVector) ? conds[i] : conds
    
#     # --- BUG FIX 2 ---
#     # Use atomics here to be safe, as A_diag is shared
#     # A[i,i] = -sum(C_internal) - C_ghb
#     Atomix.@atomic A_diag[idx] -= C # This MUST be subtraction
#     Atomix.@atomic b[idx] += C * h
# end

# @kernel function apply_flux_kernel!(b, @Const(indices), @Const(fluxes))
#     i = @index(Global, Linear)
#     idx = indices[i]
#     Q = (fluxes isa AbstractVector) ? fluxes[i] : fluxes
    
#     Atomix.@atomic b[idx] += Q
# end

# # --- CPU-compatible boundary condition functions (AD-compatible) ---
# """
# Applies General Head Boundary condition to A_diag and b on CPU.
# This is AD-compatible (no kernels, no atomics).
# """
# function _apply_ghb_cpu!(
#     A_diag::AbstractVector{T},
#     b::AbstractVector{T},
#     bc::GeneralHeadBC
# ) where T
#     indices = (bc.indices isa Vector) ? bc.indices : Array(bc.indices)
#     heads = (bc.head isa Vector) ? Array(bc.head) : bc.head
#     conds = (bc.conductance isa Vector) ? Array(bc.conductance) : bc.conductance
    
#     for i in eachindex(indices)
#         idx = indices[i]
#         h = (heads isa AbstractVector) ? heads[i] : heads
#         C = (conds isa AbstractVector) ? conds[i] : conds
        
#         # A[i,i] = -sum(C_internal) - C_ghb
#         A_diag[idx] -= C
#         b[idx] += C * h
#     end
#     return nothing
# end

# """
# Applies Flux boundary condition to b on CPU.
# This is AD-compatible (no kernels, no atomics).
# """
# function _apply_flux_cpu!(
#     b::AbstractVector{T},
#     bc::FluxBC
# ) where T
#     indices = (bc.indices isa Vector) ? bc.indices : Array(bc.indices)
#     fluxes = (bc.flux isa Vector) ? Array(bc.flux) : bc.flux

#     for i in eachindex(indices)
#         idx = indices[i]
#         Q = (fluxes isa AbstractVector) ? fluxes[i] : fluxes
#         b[idx] += Q
#     end
#     return nothing
# end
# # --- 2. CPU ASSEMBLY FUNCTION ---
# # (This is the function from the last step - it is correct and complete)
# function _assemble_matrix_cpu(
#     grid::PlanarRegularGrid,
#     Cx::Array{T, 3},
#     Cy::Array{T, 3},
#     Cz::Array{T, 3},
#     A_diag::Vector{T}
# ) where T
    
#     nlay, nrow, ncol = grid.nlay, grid.nrow, grid.ncol
#     n_nodes = nlay * nrow * ncol
    
#     # ... (Calculate nnz_total) ...
#     nnz_x_conns = nlay * nrow * (ncol - 1)
#     nnz_y_conns = nlay * (nrow - 1) * ncol
#     nnz_z_conns = (nlay - 1) * nrow * ncol
#     nnz_off_diag = 2 * (nnz_x_conns + nnz_y_conns + nnz_z_conns)
#     nnz_diag = n_nodes
#     nnz_total = nnz_off_diag + nnz_diag
    
#     I = Vector{Int}(undef, nnz_total)
#     J = Vector{Int}(undef, nnz_total)
#     V = Vector{T}(undef, nnz_total)
    
#     k = 1 
    
#     # --- Fill Off-Diagonal Entries (from Cx, Cy, Cz) ---
#     for l in 1:nlay, r in 1:nrow, c in 1:ncol
#         idx = _to_linear_index(grid, l, r, c)

#         # Off-diagonal entries should be *positive*
#         if c < ncol
#             C = Cx[l, r, c]; n_idx = _to_linear_index(grid, l, r, c + 1)
#             # --- THE FIX ---
#             # Remove the complex check.
#             # The Cx, Cy, Cz arrays are already 0.0 for inactive connections.
#             # sparse() will handle (or drop) these zero entries.
#             I[k] = idx;   J[k] = n_idx; V[k] = C; k += 1
#             I[k] = n_idx; J[k] = idx;   V[k] = C; k += 1
#         end
#         if r < nrow
#             C = Cy[l, r, c]; n_idx = _to_linear_index(grid, l, r + 1, c)
#             I[k] = idx;   J[k] = n_idx; V[k] = C; k += 1
#             I[k] = n_idx; J[k] = idx;   V[k] = C; k += 1
#         end
#         if l < nlay
#             C = Cz[l, r, c]; n_idx = _to_linear_index(grid, l + 1, r, c)
#             I[k] = idx;   J[k] = n_idx; V[k] = C; k += 1
#             I[k] = n_idx; J[k] = idx;   V[k] = C; k += 1
#         end
#     end
    
#     # --- Fill Diagonal Entries ---
#     # A_diag is now the correct negative sum, or 1.0 for inactive
#     for i in 1:n_nodes
#         I[k] = i; J[k] = i; V[k] = A_diag[i]; k += 1
#     end
    
#     # Trim unused entries (if any connections were skipped)
#     # This is no longer necessary, as k-1 should equal nnz_total
#     # But we leave it in case we add logic to skip zero-C entries
#     I_trim = I[1:k-1]
#     J_trim = J[1:k-1]
#     V_trim = V[1:k-1]

#     # `sparse` will sum duplicate entries (like the diagonal)
#     return sparse(I_trim, J_trim, V_trim, n_nodes, n_nodes)
# end

# # --- 3. CONSTANT HEAD BC (CPU-side) ---
# """
# Applies ConstantHeadBC (Dirichlet) to the *assembled* system.
# This modifies A and b *in place*.

# This is a new, more robust implementation.
# """
# function _apply_chb_cpu!(
#     A::SparseMatrixCSC, 
#     b::AbstractVector, # Can be CPU or GPU vector
#     model::FlowModel
# )
#     # --- 1. Collect all unique CHB nodes and their heads ---
#     chb_map = Dict{Int, eltype(b)}()

#     for bc_name in keys(model.conditions)
#         bc = model.conditions[bc_name]
#         if !(bc isa ConstantHeadBC)
#             continue
#         end
        
#         indices = (bc.indices isa Vector) ? bc.indices : Array(bc.indices)
#         heads   = (bc.head isa Vector) ? Array(bc.head) : bc.head

#         if heads isa AbstractVector
#             for (i, idx) in enumerate(indices)
#                 chb_map[idx] = heads[i] # Last one wins
#             end
#         else
#             for idx in indices
#                 chb_map[idx] = heads # Broadcast single value
#             end
#         end
#     end
    
#     if isempty(chb_map)
#         return Int[] # No CHB nodes
#     end
    
#     chb_indices = collect(keys(chb_map))
#     chb_set = Set(chb_indices)
    
#     # --- 2. Modify system for each unique CHB node ---
    
#     # Get direct access to the sparse matrix data
#     rows = rowvals(A)
#     vals = nonzeros(A)
    
#     # Loop over all columns
#     for j in 1:size(A, 2)
#         # Loop over all non-zero entries in this column
#         for k in A.colptr[j] : (A.colptr[j+1] - 1)
#             i = rows[k] # This is the row index A[i, j]
            
#             # Check if either the row or the column is a CHB node
#             is_chb_row = i in chb_set
#             is_chb_col = j in chb_set
            
#             if i == j
#                 # This is a diagonal entry.
#                 # If it's a CHB node, we'll set it to 1.0 later.
#                 # If it's a neighbor, we don't touch it.
#                 continue
            
#             elseif is_chb_col
#                 # This is A[i, j] where `j` is a CHB node
#                 # This entry is in a CHB *column*.
                
#                 # We must modify the `b` vector of the neighbor `i`
#                 h_fixed = chb_map[j]
#                 A_val = vals[k] # This is A[i, j], which is -C
                
#                 # b[i] = b[i] - A[i, j] * h_fixed
#                 b[i] -= A_val * h_fixed
                
#                 # Zero out the matrix entry
#                 vals[k] = 0.0
                
#             elseif is_chb_row
#                 # This is A[i, j] where `i` is a CHB node
#                 # This entry is in a CHB *row*.
#                 # We just zero it out. The `b` vector for row `i`
#                 # will be overwritten entirely.
#                 vals[k] = 0.0
#             end
#         end
#     end
    
#     # --- 3. Set diagonal to 1.0 and RHS to fixed head ---
#     # We must do this *after* the loop above,
#     # because A[idx, idx] = 1.0 is a slow operation
#     # that can re-allocate the sparse matrix data.
#     for idx in chb_indices
#         A[idx, idx] = 1.0
#         b[idx] = chb_map[idx]
#     end
    
#     dropzeros!(A) # Clean up the matrix
#     return chb_indices
# end


# # --- 4. THE MAIN ORCHESTRATOR FUNCTION ---

# function build_system(model::FlowModel{<:PlanarRegularGrid}, backend)
#     grid = model.grid
#     nlay, nrow, ncol = grid.nlay, grid.nrow, grid.ncol
#     n_nodes = nlay * nrow * ncol
#     T = eltype(grid.delr)

#     # Get one of the arrays from the model (e.g., k_horiz)
#     # This is our "source of truth" for the device and type
#     props_array = model.properties.k_horiz

#     # Get the backend (e.g., CPU() or CUDABackend()) from the array
#     backend = KernelAbstractions.get_backend(props_array)
    
#     # Get the element type (e.g., Float64) from the array
#     T = eltype(props_array)
    
#     # --- 1. Get Cache (Allocate or Reuse) ---
#     cache = get_solver_cache(model, backend)
    
#     # Reset arrays (Zero out)
#     fill!(cache.Cx, zero(T))
#     fill!(cache.Cy, zero(T))
#     fill!(cache.Cz, zero(T))
#     fill!(cache.A_diag, zero(T))
#     fill!(cache.b, zero(T))
    
#     # --- 2. Launch Kernels ---
#     k1 = compute_conductances_kernel!(backend)(
#         cache.Cx, cache.Cy, cache.Cz, grid, 
#         model.properties.k_horiz, model.properties.k_vert;
#         ndrange=(nlay, nrow, ncol)
#     )
#     k2 = build_diag_rhs_kernel!(backend)(
#         cache.A_diag, cache.b, cache.Cx, cache.Cy, cache.Cz, grid;
#         ndrange=(nlay, nrow, ncol)
#     )
    
#     synchronize(backend) # Wait for kernels to finish

#     # --- 3. Gather Results to CPU ---
#     Cx_cpu = Array(cache.Cx)
#     Cy_cpu = Array(cache.Cy)
#     Cz_cpu = Array(cache.Cz)
#     A_diag_cpu = Array(cache.A_diag)
#     b_cpu = Array(cache.b)
    
#     # --- 4. Apply BCs using CPU functions (AD-compatible) ---
#     for bc_name in keys(model.conditions)
#         bc = model.conditions[bc_name]
        
#         if bc isa FluxBC
#             _apply_flux_cpu!(b_cpu, bc)
#         elseif bc isa GeneralHeadBC
#             _apply_ghb_cpu!(A_diag_cpu, b_cpu, bc)
#         end
#     end

#     # --- 5. Assemble matrix on CPU ---
#     A_cpu = _assemble_matrix_cpu(grid, Cx_cpu, Cy_cpu, Cz_cpu, A_diag_cpu)
    
#     # --- 6. Apply Constant Head BCs (CPU-side) ---
#     chb_indices = _apply_chb_cpu!(A_cpu, b_cpu, model)
    
#     return A_cpu, b_cpu, chb_indices
# end


# function build_rhs(model, backend)

#     function rhs!(dheads, heads, p, t)
#     end
#     return rhs!
# end