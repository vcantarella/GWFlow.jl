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
    ValVec<:AbstractVector{T}
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
    ValVec<:AbstractVector{T}
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
    ValH<:AbstractVector{T},
    ValC<:AbstractVector{T}
} <: BoundaryCondition{T}
    
    "Linear indices of the nodes"
    indices::IdxVec
    
    "External head (can be a single value or an array)"
    head::ValH
    
    "Conductance from the node to the external head (can be single or array)"
    conductance::ValC
end

"""
RiverBC (Robin / Mixed / Nonlinear)
Non linear river boundary condition that accounts when aquifer heads are below aquifer bottom.
(e.g., river)
"""
struct RiverBC{
    T,
    IdxVec<:AbstractVector{Int},
    ValH<:AbstractVector{T},
    ValC<:AbstractVector{T},
    ValB<:AbstractVector{T},
} <: BoundaryCondition{T}
    
    "Linear indices of the nodes"
    indices::IdxVec
    
    "External stage (can be a single value or an array)"
    stage::ValH
    
    "Conductance from the node to the external head (can be single or array)"
    conductance::ValC

    "River bed height"
    bottom::ValB
end

"""
DrainBC (Robin / Mixed / Nonlinear)
Non linear drain boundary condition that
only withdraw water when aquifer heads are above drain height.
(e.g., streams, springs, drainage pipes)
"""
struct DrainBC{
    T,
    IdxVec<:AbstractVector{Int},
    ValH<:AbstractVector{T},
    ValC<:AbstractVector{T},
} <: BoundaryCondition{T}
    
    "Linear indices of the nodes"
    indices::IdxVec
    
    "External stage (can be a single value or an array)"
    stage::ValH
    
    "Conductance from the node to the external head (can be single or array)"
    conductance::ValC
end

struct EvapotranspirationBC{
    T,
    IdxVec<:AbstractVector{Int},
    ValS<:AbstractVector{T},
    ValD<:AbstractVector{T},
    ValE<:AbstractVector{T},
} <: BoundaryCondition{T}
    
    "Linear indices of the nodes"
    indices::IdxVec
    
    "Surface (evaporation surface)"
    surface::ValS

    "Extincion depth"
    ext_depth::ValD

    "Evapotranspiration rate"
    evt::ValE
    
end


# --- 3. INTERNAL HELPER FUNCTIONS ---

"""
Converts a 3D (layer, row, col) index to a 1D linear index.
Wraps the grid's canonical indexing method.
"""
function _to_linear_index(grid::PlanarRegularGrid, l::Int, r::Int, c::Int)
    return GWGrids.get_linear_index(grid, l, r, c)
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
    data::Union{T, AbstractVector{T}},
    id_vec::AbstractVector
) where {K,T}
    
    # Get the grid's array type (e.g. Vector or CuArray)
    ArrayType = typeof(grid.delr)

    return if data isa AbstractVector
        # If it's a vector, ensure it's the correct ArrayType
        ArrayType(T.(data))
    else
        # It's a single value, we need to repeat and match the locations vec
        ArrayType(repeat([T(data)],length(id_vec)))
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
    flux_val = _prepare_bc_data(grid, Q, indices_vec)

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
    flux_data = _prepare_bc_data(grid, flux, indices_vec)

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
    head_data = _prepare_bc_data(grid, head, indices_vec)

    #T = eltype(head_data)

    # 3. Return the generic ConstantHeadBC struct
    return ConstantHeadBC(indices_vec, head_data)
end


"""
    GeneralHeadBC(grid, locations, head, conductance)

Creates a General Head Boundary (GHB) condition.
Flow is calculated as \$Q = C (h_{ext} - h_{aquifer})\$.

# Arguments
- `grid::PlanarRegularGrid`: The grid object.
- `locations::AbstractVector{<:Tuple{Int, Int, Int}}`: List of `(l, r, c)` tuples.
- `head::Union{T, AbstractVector{T}}`: The external head (L).
- `conductance::Union{T, AbstractVector{T}}`: The conductance (L²/T).
"""
function GeneralHeadBC(grid::PlanarRegularGrid,
    locations::AbstractVector{<:Tuple{Int, Int, Int}},
    head::Union{<:Real, AbstractVector{<:Real}},
    conductance::Union{<:Real, AbstractVector{<:Real}}
    )
    # 1. Convert (l, r, c) tuples to linear indices
    indices_vec = _to_linear_indices(grid, locations)
    
    # 2. Prepare data
    stage_data = _prepare_bc_data(grid, head, indices_vec)
    cond_data = _prepare_bc_data(grid, conductance, indices_vec)
    
    # 3. Return the generic GeneralHeadBC struct
    return GeneralHeadBC(indices_vec, stage_data, cond_data)
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
    bottom::Union{<:Real, AbstractVector{<:Real}},
    conductance::Union{<:Real, AbstractVector{<:Real}}
)
    
    # 1. Convert (l, r, c) tuples to linear indices
    indices_vec = _to_linear_indices(grid, locations)
    
    # 2. Prepare data
    stage_data = _prepare_bc_data(grid, stage, indices_vec)
    cond_data = _prepare_bc_data(grid, conductance, indices_vec)
    bottom_data = _prepare_bc_data(grid, bottom, indices_vec)
    
    # 3. Return the RiverBC struct
    return RiverBC(indices_vec, stage_data, cond_data, bottom_data)
end

"""
    RechargeBC(grid, locations, recharge_rate)

Applies areal recharge to the top layer of the model.
Automatically converts the recharge rate (L/T) to a volumetric flux (L³/T)
based on the cell area.

# Arguments
- `grid::PlanarRegularGrid`: The grid object.
- `locations::AbstractVector{<:Tuple{Int, Int}}`: List of `(row, col)` tuples (2D coordinates).
- `recharge_rate::Union{T, AbstractVector{T}}`: The recharge rate (L/T).
  Positive values indicate water entering the aquifer.
"""
function RechargeBC(
    grid::PlanarRegularGrid,
    locations::AbstractVector{<:Tuple{Int, Int}},
    recharge_rate::Union{<:Real, AbstractVector{<:Real}}
)
    # 1. Convert (r, c) tuples to (1, r, c) linear indices 
    up_locations = [(1, r, c) for (r,c) in locations]
    indices_vec = _to_linear_indices(grid, up_locations)
    
    areas = [grid.delr[c]*grid.delc[r] for (r,c) in locations]
    flux = recharge_rate.*areas
    # 2. Prepare flux data (move to GPU if needed)
    flux_data = _prepare_bc_data(grid, flux, indices_vec)

    # 3. Return the generic FluxBC struct
    return FluxBC(indices_vec, flux_data)
end

"""
    DrainBC(grid, locations, stage, conductance)

Creates a Drain boundary condition.
Acts like a General Head Boundary but only extracts water when \$h_{aquifer} > h_{stage}\$.

# Arguments
- `grid::PlanarRegularGrid`: The grid object.
- `locations::AbstractVector{<:Tuple{Int, Int, Int}}`: List of `(l, r, c)` tuples.
- `stage::Union{T, AbstractVector{T}}`: The drain elevation/stage (L).
- `conductance::Union{T, AbstractVector{T}}`: The drain conductance (L²/T).
"""
function DrainBC(grid::PlanarRegularGrid,
    locations::AbstractVector{<:Tuple{Int, Int, Int}},
    stage::Union{<:Real, AbstractVector{<:Real}},
    conductance::Union{<:Real, AbstractVector{<:Real}}
    )
    # 1. Convert (l, r, c) tuples to linear indices
    indices_vec = _to_linear_indices(grid, locations)
    
    # 2. Prepare data
    stage_data = _prepare_bc_data(grid, stage, indices_vec)
    cond_data = _prepare_bc_data(grid, conductance, indices_vec)
    
    # 3. Return the DrainBC struct
    return DrainBC(indices_vec, stage_data, cond_data)
end

function EvapotranspirationBC(grid::PlanarRegularGrid,
    locations::AbstractVector{<:Tuple{Int, Int, Int}},
    surface::Union{<:Real, AbstractVector{<:Real}},
    ext_depth::Union{<:Real, AbstractVector{<:Real}},
    evt::Union{<:Real, AbstractVector{<:Real}}
    )
    # 1. Convert (l, r, c) tuples to linear indices
    indices_vec = _to_linear_indices(grid, locations)
    
    # 2. Prepare data
    surface_data = _prepare_bc_data(grid, surface, indices_vec)
    ext_depth_data = _prepare_bc_data(grid, ext_depth, indices_vec)
    evt_data = _prepare_bc_data(grid, evt, indices_vec)
    # 3. Return the DrainBC struct
    return EvapotranspirationBC(indices_vec, surface_data, ext_depth_data, evt_data)
end

@inline function bc_priority(::FluxBC) 1 end
@inline function bc_priority(::GeneralHeadBC) 1 end
@inline function bc_priority(::ConstantHeadBC) 1 end
@inline function bc_priority(::RiverBC) 1 end
@inline function bc_priority(::DrainBC) 1 end
@inline function bc_priority(::EvapotranspirationBC) 1 end
Base.isless(a::BoundaryCondition, b::BoundaryCondition) = bc_priority(a) < bc_priority(b)