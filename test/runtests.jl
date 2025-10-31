using GWFlow
using Test

@testset "GWFlow Tests" begin
    @testset "solve" begin
        include("test_olsthoorn.jl")
    end
end