using GWSolve
using Test

@testset "GWSolve Tests" begin
    @testset "solve" begin
        @test "simple case" begin
            result = GWSolve.solve(1, 2, 3)
            @test result == 6
        end
    end
end