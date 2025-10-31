using GWGrids
using SparseArrays
using LinearSolve


struct FlowModel{G<:PlanarRegularGrid, P, BC, SCT}
    grid::G
    
    # Hydraulic properties (K, porosity, etc.)
    # A NamedTuple of arrays: (k=K_array, n=porosity_array)
    properties::P 
    
    # Boundary conditions, wells, etc.
    # A tuple or vector of BC structs
    conditions::BC
    
    # Configuration for the solver
    solver_config::SCT # e.g., (algorithm=KrylovJL_CG(), abstol=1e-6)
end

struct FlowSolution{
    T<:AbstractFloat, 
    ArrT3D<:AbstractArray{T, 3}, 
    G<:PlanarRegularGrid{T},
    BCR<:NamedTuple  # <--- NEW TYPE PARAMETER
}
    grid::G
    head::ArrT3D
    flows::NamedTuple{(:right, :front, :lower), NTuple{3, ArrT3D}}
    
    # --- NEW FIELD ---
    # Stores the flow results, matching the names from model.conditions
    bc_flows::BCR
end