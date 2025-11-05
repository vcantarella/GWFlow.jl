# --- 1. ABSTRACT TYPE ---
abstract type BoundaryCondition{T} end


# --- 2. GENERIC STRUCT DEFINITIONS (for the solver) ---

"""
ConstantHeadBC (Dirichlet)
Applies a fixed head value to a set of nodes.
"""
struct ConstantHeadBC{
    T,
    IdxVec<:AbstractVector{Int},
    ValVec<:Union{T, AbstractVector{T}}
} <: BoundaryCondition{T}
    
    "Linear indices of the nodes"
    indices::IdxVec
    
    "Head values (can be a single value or an array of values)"
    head::ValVec
end

"""
FluxBC (Neumann)
Applies a fixed flux (e..g, a well) to a set of nodes.
Positive flux is injection, negative flux is extraction.
"""
struct FluxBC{
    T,
    IdxVec<:AbstractVector{Int},
    ValVec<:Union{T, AbstractVector{T}}
} <: BoundaryCondition{T}
    
    "Linear indices of the nodes"
    indices::IdxVec
    
    "Flux values (can be a single value or an array)"
    flux::ValVec
end

"""
GeneralHeadBC (Robin / Mixed)
Connects nodes to an external head via a conductance.
(e.g., river, lake, drain)
"""
struct GeneralHeadBC{
    T,
    IdxVec<:AbstractVector{Int},
    ValH<:Union{T, AbstractVector{T}},
    ValC<:Union{T, AbstractVector{T}}
} <: BoundaryCondition{T}
    
    "Linear indices of the nodes"
    indices::IdxVec
    
    "External head (can be a single value or an array)"
    head::ValH
    
    "Conductance from the node to the external head (can be single or array)"
    conductance::ValC
end


# --- 3. INTERNAL HELPER FUNCTIONS ---

"""
Converts a 3D (layer, row, col) index to a 1D linear index.
Assumes array storage is (nlay, nrow, ncol).
"""
function _to_linear_index(grid::PlanarRegularGrid, l::Int, r::Int, c::Int)
    # Perform bounds checking
    if !(1 <= l <= grid.nlay && 1 <= r <= grid.nrow && 1 <= c <= grid.ncol)
        error("Grid index ($l, $r, $c) is out of bounds for grid size ($grid.nlay, $grid.nrow, $grid.ncol)")
    end
    
    # Calculate linear index for A[l, r, c]
    linear_idx = (c - 1) * grid.nrow * grid.nlay + (r - 1) * grid.nlay + l
    return linear_idx
end

"""
Converts a list of 3D (l, r, c) indices to a 1D linear index vector.
This function is GPU-aware.
"""
function _to_linear_indices(
    grid::PlanarRegularGrid, 
    locations::AbstractVector{<:Tuple{Int, Int, Int}}
)
    n_nodes = length(locations)
    n_nodes = length(locations)

    # Create an output container with the same device/container type as grid.delr
    # but with element type Int and length n_nodes. Using the grid's array as
    # a prototype ensures CPU/GPU compatibility (e.g., Vector vs CuArray).
    indices_vec = similar(grid.delr, Int, n_nodes)

    # Ensure we iterate over a CPU-side Vector of location tuples
    cpu_locations = locations isa Vector ? locations : collect(locations)

    # Compute indices on the CPU
    indices_cpu = Vector{Int}(undef, n_nodes)
    for (i, (l, r, c)) in enumerate(cpu_locations)
        indices_cpu[i] = _to_linear_index(grid, l, r, c)
    end

    # Copy values into the output container (works for CPU->CPU or CPU->GPU)
    copy!(indices_vec, indices_cpu)

    return indices_vec
end

"""
Helper to prepare data (Float or Vector) for the BC struct,
moving it to the correct device (CPU/GPU) to match the grid.
"""
function _prepare_bc_data(
    grid::PlanarRegularGrid{K},
    data::Union{T, AbstractVector{T}}
) where {K,T}
    
    # Get the grid's array type (e.g. Vector or CuArray)
    ArrayType = typeof(grid.delr)

    return if data isa AbstractVector
        # If it's a vector, ensure it's the correct ArrayType
        ArrayType(T.(data))
    else
        # It's a single value, just convert type
        T(data)
    end
end


# --- 4. USER-FRIENDLY ALIASES / CONSTRUCTORS ---

"""
    Well(grid, l, r, c, Q)
    Well(grid, location, Q)

Creates a single-cell flux boundary condition (a well).

# Arguments
- `grid::PlanarRegularGrid`: The grid object.
- `l, r, c`: The (layer, row, col) of the well.
- `location`: A single `(l, r, c)` tuple.
- `Q`: The flow rate (L³/T). Negative for pumping, positive for injection.
"""
function Well(
    grid::PlanarRegularGrid{T},
    l::Int, r::Int, c::Int,
    Q::Real
) where T
    
    # 1. Get the single linear index
    idx = _to_linear_index(grid, l, r, c)
    
    # 2. Create a 1-element index container with same device/type as grid.delr
    indices_vec = similar(grid.delr, Int, 1)
    indices_vec[1] = idx

    # 3. Prepare the flux value on the correct device/type
    flux_val = _prepare_bc_data(grid, Q)

    # 4. Return the generic FluxBC struct with explicit type parameters
    FT = eltype(flux_val)
    return FluxBC{FT, typeof(indices_vec), typeof(flux_val)}(indices_vec, flux_val)
end

function Well(grid::PlanarRegularGrid, location::Tuple{Int, Int, Int}, Q::Real)
    return Well(grid, location[1], location[2], location[3], Q)
end


"""
    FluxBC(grid, locations, flux)

Creates a flux boundary condition (e.g., regional flow) over one or more cells.

# Arguments
- `grid::PlanarRegularGrid`: The grid object.
- `locations::AbstractVector{<:Tuple{Int, Int, Int}}`: A list of `(l, r, c)` tuples.
- `flux::Union{T, AbstractVector{T}}`: The flux (L³/T). 
  Can be a single value (applied to all locations) or a vector matching `locations`.
"""
function FluxBC(
    grid::PlanarRegularGrid,
    locations::AbstractVector{<:Tuple{Int, Int, Int}},
    flux::Union{<:Real, AbstractVector{<:Real}}
)
    
    # 1. Convert (l, r, c) tuples to linear indices
    indices_vec = _to_linear_indices(grid, locations)
    
    # 2. Prepare flux data (move to GPU if needed)
    flux_data = _prepare_bc_data(grid, flux)

    # 3. Return the generic FluxBC struct
    return FluxBC(indices_vec, flux_data)
end


"""
    ConstantHeadBC(grid, locations, head)

Creates a constant head boundary condition over one or more cells.

# Arguments
- `grid::PlanarRegularGrid`: The grid object.
- `locations::AbstractVector{<:Tuple{Int, Int, Int}}`: A list of `(l, r, c)` tuples.
- `head::Union{T, AbstractVector{T}}`: The head value (L). 
  Can be a single value or a vector matching `locations`.
"""
function ConstantHeadBC(
    grid::PlanarRegularGrid,
    locations::AbstractVector{<:Tuple{Int, Int, Int}},
    head::Union{T, AbstractVector{T}}
) where {T}

    # 1. Convert (l, r, c) tuples to linear indices
    indices_vec = _to_linear_indices(grid, locations)
    
    # 2. Prepare head data (move to GPU if needed)
    head_data = _prepare_bc_data(grid, head)

    #T = eltype(head_data)

    # 3. Return the generic ConstantHeadBC struct
    return ConstantHeadBC(indices_vec, head_data)
end


"""
    RiverBC(grid, locations, stage, conductance)

Creates a linear River boundary condition (a form of General Head Boundary).
Assumes \$Q = C (h_{stage} - h_{aquifer})\$.

# Arguments
- `grid::PlanarRegularGrid`: The grid object.
- `locations::AbstractVector{<:Tuple{Int, Int, Int}}`: List of `(l, r, c)` tuples.
- `stage::Union{T, AbstractVector{T}}`: The river stage (L). 
  Can be a single value or a vector matching `locations`.
- `conductance::Union{T, AbstractVector{T}}`: The riverbed conductance (L²/T). 
  Can be a single value or a vector matching `locations`.
"""
function RiverBC(
    grid::PlanarRegularGrid,
    locations::AbstractVector{<:Tuple{Int, Int, Int}},
    stage::Union{<:Real, AbstractVector{<:Real}},
    conductance::Union{<:Real, AbstractVector{<:Real}}
)
    
    # 1. Convert (l, r, c) tuples to linear indices
    indices_vec = _to_linear_indices(grid, locations)
    
    # 2. Prepare data
    stage_data = _prepare_bc_data(grid, stage)
    cond_data = _prepare_bc_data(grid, conductance)
    
    # 3. Return the generic GeneralHeadBC struct
    return GeneralHeadBC(indices_vec, stage_data, cond_data)
end
