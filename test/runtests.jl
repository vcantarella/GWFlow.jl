using GWSolve
using Test

@testset "GWSolve Tests" begin
    @testset "solve" begin
        include("test_olsthoorn.jl")
    end
end