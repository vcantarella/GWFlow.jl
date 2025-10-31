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

# TODO: get condunctance between two cells
@inline function _get_conductance(K1, L1, K2, L2, Area)
    return error("Not implemented")
end

@kernel function compute_conductances_kernel!(Cx, Cy, Cz, grid, k_horiz, k_vert)
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
        K1 = k_horiz[l, r, c]
        K2 = k_horiz[l, r, c+1]
        Cx[l, r, c] = _get_conductance(K1, L1, K2, L2, Area)
    end
    
    # --- Y-direction Conductance ---
    if r < nrow
        Area = delr_c * deltaz_l
        L1 = delc_r / 2.0
        L2 = grid.delc[r+1] / 2.0
        K1 = k_horiz[l, r, c]
        K2 = k_horiz[l, r+1, c]
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

@kernel function build_diag_rhs_kernel!(
    A_diag, b,
    Cx, Cy, Cz,
    grid
)
    l, r, c = @index(Global, NTuple) # 3D thread index
    idx = _to_linear_index(grid, l, r, c) # Assuming this is GPU-safe

    nlay, nrow, ncol = grid.nlay, grid.nrow, grid.ncol
    
    diag_val = 0.0
    
    if c > 1;   diag_val += Cx[l, r, c-1]; end
    if c < ncol; diag_val += Cx[l, r, c];   end
    if r > 1;   diag_val += Cy[l, r-1, c]; end
    if r < nrow; diag_val += Cy[l, r, c];   end
    if l > 1;   diag_val += Cz[l-1, r, c]; end
    if l < nlay; diag_val += Cz[l, r, c];   end
    
    A_diag[idx] = diag_val
    b[idx] = 0.0
end

@kernel function apply_ghb_kernel!(A_diag, b, indices, heads, conds)
    i = @index(Global, Linear) # 1D thread index
    
    idx = indices[i]
    h = (heads isa AbstractVector) ? heads[i] : heads
    C = (conds isa AbstractVector) ? conds[i] : conds
    
    # Use atomics here to be safe, as A_diag is shared
    Atomix.@atomic A_diag[idx] += C
    Atomix.@atomic b[idx] += C * h
end

@kernel function apply_flux_kernel!(b, indices, fluxes)
    i = @index(Global, Linear)
    idx = indices[i]
    Q = (fluxes isa AbstractVector) ? fluxes[i] : fluxes
    
    Atomix.@atomic b[idx] += Q
end
# --- 2. CPU ASSEMBLY FUNCTION ---
# (This is the function from the last step - it is correct and complete)
function _assemble_matrix_cpu(
    grid::PlanarRegularGrid{T},
    Cx::Array{T, 3},
    Cy::Array{T, 3},
    Cz::Array{T, 3},
    A_diag::Vector{T} # <-- This A_diag *already includes* GHB conductances
) where T
    
    nlay, nrow, ncol = grid.nlay, grid.nrow, grid.ncol
    n_nodes = nlay * nrow * ncol
    
    # ... (Calculate nnz_total) ...
    nnz_x_conns = nlay * nrow * (ncol - 1)
    nnz_y_conns = nlay * (nrow - 1) * ncol
    nnz_z_conns = (nlay - 1) * nrow * ncol
    nnz_off_diag = 2 * (nnz_x_conns + nnz_y_conns + nnz_z_conns)
    nnz_diag = n_nodes
    nnz_total = nnz_off_diag + nnz_diag
    
    I = Vector{Int}(undef, nnz_total)
    J = Vector{Int}(undef, nnz_total)
    V = Vector{T}(undef, nnz_total)
    
    k = 1 
    
    # --- Fill Off-Diagonal Entries (from Cx, Cy, Cz) ---
    for l in 1:nlay, r in 1:nrow, c in 1:ncol
        idx = _to_linear_index(grid, l, r, c)
        if c < ncol
            C = Cx[l, r, c]; n_idx = _to_linear_index(grid, l, r, c + 1)
            I[k] = idx;   J[k] = n_idx; V[k] = -C; k += 1
            I[k] = n_idx; J[k] = idx;   V[k] = -C; k += 1
        end
        if r < nrow
            C = Cy[l, r, c]; n_idx = _to_linear_index(grid, l, r + 1, c)
            I[k] = idx;   J[k] = n_idx; V[k] = -C; k += 1
            I[k] = n_idx; J[k] = idx;   V[k] = -C; k += 1
        end
        if l < nlay
            C = Cz[l, r, c]; n_idx = _to_linear_index(grid, l + 1, r, c)
            I[k] = idx;   J[k] = n_idx; V[k] = -C; k += 1
            I[k] = n_idx; J[k] = idx;   V[k] = -C; k += 1
        end
    end
    
    # --- Fill Diagonal Entries ---
    # This A_diag was computed on the GPU and *already includes*
    # all internal conductances AND the GeneralHeadBC conductances.
    for i in 1:n_nodes
        I[k] = i; J[k] = i; V[k] = A_diag[i]; k += 1
    end
    
    return sparse(I, J, V, n_nodes, n_nodes)
end


# --- 3. CONSTANT HEAD BC (CPU-side) ---
"""
Applies ConstantHeadBC (Dirichlet) to the *assembled* system.
This modifies A and b *in place*.
"""
function _apply_chb_cpu!(
    A::SparseMatrixCSC, 
    b::AbstractVector, # Can be CPU or GPU vector
    model::FlowModel
)
    chb_indices = Int[] # Collect all CHB indices

    for bc_name in keys(model.conditions)
        bc = model.conditions[bc_name]
        if !(bc isa ConstantHeadBC)
            continue
        end

        # Bring indices/heads to CPU for modification
        indices = (bc.indices isa Vector) ? bc.indices : Array(bc.indices)
        heads   = (bc.head isa Vector) ? Array(bc.head) : bc.head

        for (i, idx) in enumerate(indices)
            push!(chb_indices, idx)
            h_fixed = (heads isa Vector) ? heads[i] : heads

            # Find all non-zero entries in column `idx`
            for k in A.colptr[idx] : (A.colptr[idx+1] - 1)
                neighbor_idx = A.rowval[k]
                if neighbor_idx == idx
                    continue # Skip the diagonal
                end
                
                C_val = -A.nzval[k] # Conductance
                
                # b[neighbor] = b[neighbor] - (-C * h_fixed)
                # This works even if b is a CuArray!
                b[neighbor_idx] += C_val * h_fixed
            end
            
            # Zero out column `idx`
            # (We must also zero row `idx` for symmetry)
            dropzeros!(select(rowcol -> rowcol[2] == idx || rowcol[1] == idx, A))
            
            # Set diagonal to 1 and b to fixed head
            A[idx, idx] = 1.0
            b[idx] = h_fixed
        end
    end
    
    dropzeros!(A) # Clean up
    return chb_indices
end


# --- 4. THE MAIN ORCHESTRATOR FUNCTION ---

function build_system(model::FlowModel{<:PlanarRegularGrid}, backend)
    grid = model.grid
    nlay, nrow, ncol = grid.nlay, grid.nrow, grid.ncol
    n_nodes = nlay * nrow * ncol
    T = eltype(grid.delr)

    # Get one of the arrays from the model (e.g., k_horiz)
    # This is our "source of truth" for the device and type
    props_array = model.properties.k_horiz

    # Get the backend (e.g., CPU() or CUDABackend()) from the array
    backend = KernelAbstractions.get_backend(props_array)
    
    # Get the element type (e.g., Float64) from the array
    T = eltype(props_array)
    
    # --- 1. Allocate Backend Arrays ---
    Cx = similar(props_array, T, (nlay, nrow, ncol)); Cx .= zero(T)
    Cy = similar(props_array, T, (nlay, nrow, ncol)); Cy .= zero(T)
    Cz = similar(props_array, T, (nlay, nrow, ncol)); Cz .= zero(T)
    A_diag = similar(props_array, T, n_nodes); A_diag .= zero(T)
    b = similar(props_array, T, n_nodes); b .= zero(T)
    
    
    # --- 2. Launch Kernels ---
    k1 = compute_conductances_kernel!(backend)(
        Cx, Cy, Cz, grid, 
        model.properties.k_horiz, model.properties.k_vert; 
        ndrange=(nlay, nrow, ncol)
    )
    k2 = build_diag_rhs_kernel!(backend)(
        A_diag, b, Cx, Cy, Cz, grid; 
        ndrange=(nlay, nrow, ncol), depends=k1
    )
    
    # Chain BC kernels
    events = [k2]
    for bc_name in keys(model.conditions)
        bc = model.conditions[bc_name]
        
        # --- BC HANDLING ---
        if bc isa FluxBC
            # Modifies `b` on device
            k_flux = apply_flux_kernel!(backend)(
                b, bc.indices, bc.flux;
                ndrange=length(bc.indices), depends=events[end]
            )
            push!(events, k_flux)
        elseif bc isa GeneralHeadBC
            # Modifies `A_diag` and `b` on device
            k_ghb = apply_ghb_kernel!(backend)(
                A_diag, b, bc.indices, bc.head, bc.conductance;
                ndrange=length(bc.indices), depends=events[end]
            )
            push!(events, k_ghb)
        end
    end
    
    wait(events[end]) # Wait for all kernels to finish
    
    # --- 3. Gather Results and Assemble on CPU ---
    Cx_cpu = Array(Cx)
    Cy_cpu = Array(Cy)
    Cz_cpu = Array(Cz)
    A_diag_cpu = Array(A_diag) # This A_diag_cpu NOW INCLUDES GHB conductances
    
    # `b` stays on the device (e.g., GPU)
    
    # This call assembles A using the GHB-modified A_diag_cpu
    A_cpu = _assemble_matrix_cpu(grid, Cx_cpu, Cy_cpu, Cz_cpu, A_diag_cpu)
    
    # --- 4. Apply Constant Head BCs (CPU-side) ---
    # --- BC HANDLING (Special Case) ---
    # This modifies A_cpu and `b` (on-device) in-place
    chb_indices = _apply_chb_cpu!(A_cpu, b, model)
    
    return A_cpu, b, chb_indices
end
