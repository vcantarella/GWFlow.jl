module GWSolve

include("grids.jl")
include("bcs.jl")
include("solve.jl")
function solve(a, b, c)
    return a + b + c
end

end