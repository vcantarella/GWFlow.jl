using Test
using KernelAbstractions
using Metal

# --- 1. Define a Functor (Recommended) ---
struct LinearTransform{T}
    slope::T
    intercept::T
end

# Make the struct callable so it acts like a function
# Use @inline to ensure it can be inlined into the kernel
@inline (f::LinearTransform)(x) = f.slope * x + f.intercept

# --- 2. Define the Kernel ---
@kernel function apply_function_kernel!(output, input, func)
    I = @index(Global)
    @inbounds output[I] = func(input[I])
end

@testset "KernelAbstractions Function Arguments" begin
    # Test on multiple backends if available
    backends = Any[CPU()]
    if Metal.functional()
        push!(backends, MetalBackend())
    end

    for backend in backends
        @testset "Testing on $(typeof(backend))" begin
            N = 100
            
            # Prepare data
            input_cpu = collect(Float32, 1.0:N)
            output_cpu = zeros(Float32, N)
            
            # Move data to the appropriate device
            if backend isa MetalBackend
                input = MtlArray(input_cpu)
                output = MtlArray(output_cpu)
            else
                input = input_cpu
                output = output_cpu
            end
            
            # --- Case A: Using a Functor ---
            transform = LinearTransform(2.0f0, 5.0f0) 
            
            kernel! = apply_function_kernel!(backend)
            kernel!(output, input, transform, ndrange=N)
            KernelAbstractions.synchronize(backend)
            
            # Copy back to CPU for verification
            res = Array(output)
            @test res ≈ (2.0f0 .* input_cpu .+ 5.0f0)
            
            # --- Case B: Using a Pure Function ---
            if backend isa MetalBackend
                output_fn = MtlArray(zeros(Float32, N))
            else
                output_fn = zeros(Float32, N)
            end
            
            # Note: passing global functions to GPU can sometimes be tricky
            # depending on the compiler's ability to see the implementation.
            # Built-in math functions usually work.
            kernel!(output_fn, input, cos, ndrange=N)
            KernelAbstractions.synchronize(backend)
            
            res_fn = Array(output_fn)
            @test res_fn ≈ cos.(input_cpu)
        end
    end
end