using Test
using LinearAlgebra
using SparseArrays
import KernelAbstractions as KA
using LinearSolve

# --- Import your packages ---
# (You must make sure GWGrids.jl and GroundwaterFlow.jl are
# available in your Julia environment)
using GWGrids
using GWFlow



# ====================================================================
#  STEP 1: THE REFERENCE SOLVER
#  (This is the script you provided, wrapped in a function)
# ====================================================================

function run_reference_solver()
    @info "Running reference 'plain Julia' solver..."

    # Specify a rectangular grid
    x = -1000.0:25.0:1000.0
    y = -1000.0:25.0:1000.0
    z = -100.0:20.0:0.0

    # Get number of cells along each axis
    Nx = length(x) - 1
    Ny = length(y) - 1
    Nz = length(z) - 1

    sz = (Nz, Ny, Nx)
    Nod = prod(sz)

    # Cell dimensions
    dx = reshape(diff(x), 1, 1, Nx)
    dy = reshape(diff(y), 1, Ny, 1)
    dz = reshape(abs.(diff(z)), Nz, 1, 1)

    # IBOUND array
    IBOUND = ones(Int, sz)
    IBOUND[:, end, :] .= -1      # Last row has prescribed heads
    IBOUND[:, 41:45, 21:70] .= 0 # Inactive cells

    active = reshape(IBOUND .> 0, Nod)
    inact = reshape(IBOUND .== 0, Nod)
    fxhd = reshape(IBOUND .< 0, Nod)

    # Hydraulic conductivities
    k = 10.0
    kx = k * ones(sz)
    ky = k * ones(sz)
    kz = k * ones(sz)

    # Half cell flow resistances
    Rx = 0.5 * dx ./ (dy .* dz) ./ kx
    Ry = 0.5 * dy ./ (dz .* dx) ./ ky
    Rz = 0.5 * dz ./ (dx .* dy) ./ kz

    # Set inactive resistances
    Rx_flat = reshape(Rx, Nod); Rx_flat[inact] .= Inf; Rx = reshape(Rx_flat, sz)
    Ry_flat = reshape(Ry, Nod); Ry_flat[inact] .= Inf; Ry = reshape(Ry_flat, sz)
    Rz_flat = reshape(Rz, Nod); Rz_flat[inact] .= Inf; Rz = reshape(Rz_flat, sz)

    # Conductances
    Cx = 1 ./ (Rx[:, :, 1:end-1] .+ Rx[:, :, 2:end])
    Cy = 1 ./ (Ry[:, 1:end-1, :] .+ Ry[:, 2:end, :])
    Cz = 1 ./ (Rz[1:end-1, :, :] .+ Rz[2:end, :, :])

    # Node numbering (0-indexed)
    NOD = reshape(0:Nod-1, sz)

    # Neighbor identification (0-indexed)
    IE = NOD[:, :, 2:end]; IW = NOD[:, :, 1:end-1]
    IN = NOD[:, 1:end-1, :]; IS = NOD[:, 2:end, :]
    IT = NOD[1:end-1, :, :]; IB = NOD[2:end, :, :]

    # Build sparse system matrix (1-indexed)
    row_indices = vcat(vec(IE), vec(IW), vec(IN), vec(IS), vec(IB), vec(IT)) .+ 1
    col_indices = vcat(vec(IW), vec(IE), vec(IS), vec(IN), vec(IT), vec(IB)) .+ 1
    values = vcat(vec(Cx), vec(Cx), vec(Cy), vec(Cy), vec(Cz), vec(Cz))
    A = sparse(row_indices, col_indices, values, Nod, Nod)

    adiag = -vec(sum(A, dims=2))
    A_diag = sparse(1:Nod, 1:Nod, adiag, Nod, Nod)
    A_complete = A + A_diag

    # Boundary conditions
    FQ = zeros(sz)
    FQ[3, 31, 26] = -1200.0 # Extraction

    HI = zeros(sz) # Initial/Fixed heads are 0.0

    # Right-hand side
    RHS = vec(FQ) - A_complete[:, fxhd] * vec(HI)[fxhd]

    # Solve
    Phi = vec(HI)
    Phi[active] = A_complete[active, active] \ RHS[active]
    Phi[inact] .= NaN # Not strictly needed, but good for plotting

    return vec(Phi), active, fxhd, inact
end


# ====================================================================
#  STEP 2: YOUR GWFlow.jl SOLVER
#  (This builds the identical model using your package's API)
# ====================================================================

function run_gwflow_solver(backend)
    @info "Running GroundwaterFlow.jl solver..."

    # --- 1. Grid Definition ---
    nlay = 5
    nrow = 80
    ncol = 80

    delr = 25.0
    delc = 25.0
    top = 0.0
    # Create layer thicknesses
    layer_thickness = 20.0

    grid = PlanarRegularGrid(
            nlay, nrow, ncol,
            delr, delc, 
            layer_thickness,
            top,
            origin=(-1000.0, -1000.0),
            angrot=0.0
        )

    # --- 2. Properties ---
    k_val = 10.0
    T = Float64

    k_horiz_arr = fill(T(k_val), (nlay, nrow, ncol))
    k_vert_arr = fill(T(k_val), (nlay, nrow, ncol))

    # Apply inactive zone by setting K=0
    # Note: `KA.zeros` is the right way to do this
    inactive_patch = zeros(T, nlay, 5, 50)
    k_horiz_arr[:, 41:45, 21:70] = inactive_patch
    k_vert_arr[:, 41:45, 21:70] = inactive_patch

    properties = (
        k_horiz = k_horiz_arr,
        k_vert = k_vert_arr
    )

    # --- 3. Boundary Conditions ---
    
    # Fixed Head (last row, head=0.0)
    chb_locs = [(l, nrow, c) for l in 1:nlay, c in 1:ncol]
    chb = GWFlow.ConstantHeadBC(grid, vec(chb_locs), T(0.0))

    # Well (Flux)
    well = GWFlow.Well(grid, 3, 31, 26, T(-1200.0))

    conditions = (
        fixed_head_N = chb,
        extraction_well = well
    )
    solver_config = (algorithm=KrylovJL_CG(), abstol=1e-6)

    # --- 4. Create Model ---
    model = GWFlow.FlowModel(grid, properties, conditions, solver_config)
    
    # --- 5. Build and Solve System ---
    (A, b, chb_indices) = GWFlow.build_system(model, backend)
    active_indices = .!(isnan.(b))
    As = A[active_indices, active_indices]
    bs = b[active_indices]
    prob = LinearProblem(As, bs)
    sol = solve(prob) # Use default sparse CPU solver
    return sol.u
end


# ====================================================================
#  STEP 3: THE TESTSET
# ====================================================================

@testset "Olsthoorn (2021) Verification Test" begin
    
    # --- Run both solvers ---
    phi_ref, active_nodes, fxhd_nodes, inact_nodes = run_reference_solver()
    phi_gwflow = run_gwflow_solver(KA.CPU())
    
    Nod = length(phi_ref)
    
    # --- Comparison ---
    # Note: phi_gwflow contains only active + fixed head nodes (no inactive nodes)
    # We need to extract the same subset from phi_ref for comparison
    
    valid_nodes = active_nodes .| fxhd_nodes  # All nodes that are not inactive
    phi_ref_valid = phi_ref[valid_nodes]
    
    @testset "Solution Size" begin
        @test length(phi_gwflow) == sum(valid_nodes)
        @test length(phi_gwflow) == sum(active_nodes) + sum(fxhd_nodes)
    end
    
    @testset "Full Solution Comparison" begin
        # Compare all valid (active + fixed head) nodes
        max_diff = maximum(abs.(phi_gwflow .- phi_ref_valid))
        @info "Maximum difference between solutions: $max_diff"
        @test phi_gwflow ≈ phi_ref_valid atol = 1e-6
    end
end