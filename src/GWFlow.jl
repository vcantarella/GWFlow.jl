module GWFlow
    using KernelAbstractions
    const KA = KernelAbstractions
    include("regular_grids.jl")
    include("boundary_conditions.jl")
    include("solver_v2.jl")
end