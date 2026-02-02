using Test
using LinearAlgebra
using SparseArrays
import KernelAbstractions as KA
using ADTypes
using SparseConnectivityTracer
detector = TracerSparsityDetector()
using GWGrids
using GWFlow
using NonlinearSolve

# ====================================================================
#  STEP 1: THE REFERENCE SOLVER (Exact Copy)
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
    FQ[3, 31, 26] = 1200.0 # Extraction

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
#  STEP 2: GWFlow.jl V2 SOLVER
# ====================================================================

function run_gwflow_solver_v2(backend)
    @info "Running GroundwaterFlow.jl V2 solver..."

    # --- 1. Grid Definition ---
    nlay, nrow, ncol = 5, 80, 80
    delr, delc = 25.0, 25.0
    grid = PlanarRegularGrid(nlay, nrow, ncol, delr, delc, 20.0, 0.0, origin=(-1000.0, -1000.0))

    # --- 2. Properties (Closure) ---
    k_val = 10.0
    k_horiz_arr = fill(k_val, (nlay, nrow, ncol))
    k_vert_arr = fill(k_val, (nlay, nrow, ncol))
    
    # Inactive zone (K=0)
    inactive_patch = zeros(nlay, 5, 50)
    k_horiz_arr[:, 41:45, 21:70] = inactive_patch
    k_vert_arr[:, 41:45, 21:70] = inactive_patch

    # Create the property accessor function
    # Returns (k_horiz, k_vert) for a given (l, r, c)
    # The kernel expects _get_k(l, r, c, p) returning (k_h, k_v) or just k if isotropic?
    # Checking solver_v2.jl: 
    # K1 = _get_k(l, r, c, p)
    # cx_left = _get_conductance(K1, ...)
    # It expects a single K value for the connection.
    # Actually, solver_v2 uses separate k_row, k_col in the conductance kernel,
    # but the NEW divergence kernel takes `_get_k`.
    # Let's assume _get_k returns the hydraulic conductivity at that cell.
    # For anisotropic, we might need direction... but the current `compute_divergence_kernel`
    # calls `K2 = _get_k(l, r, c+1, p)` for X flux. This implies K is isotropic or scalar per cell.
    # We will return the horizontal K.
    
    _get_k = (l, r, c, p, n) -> begin
        # Bounds check handled by caller or array indexing
        return k_horiz_arr[l, r, c]
    end

    # --- 3. Boundary Conditions ---
    BCs = GWFlow.BoundaryCondition[]

    # Fixed Head (last row)
    chb_locs = [(l, nrow, c) for l in 1:nlay, c in 1:ncol]
    push!(BCs, GWFlow.ConstantHeadBC(grid, vec(chb_locs), 0.0))

    # Well
    push!(BCs, GWFlow.Well(grid, 3, 31, 26, -1200.0))

    # --- 4. Build RHS Function ---
    rhs! = GWFlow.make_rhs(grid, BCs, _get_k)
    local_rhs!(du, u, p) = rhs!(du, u, p, 0.0)

    # --- 5. Solve ---
    u0 = zeros(nlay * nrow * ncol)
    # fix at chb_locs
    p = [1.0] # No dynamic parameters for now
    du0 = zeros(size(u0))
    jac_sparsity = ADTypes.jacobian_sparsity(
    (du, u) -> rhs!(du, u, p, 0.0), du0, u0, detector)
    rhs!(du0, u0, p, 0)
    # check allocations of rhs!
    @allocated rhs!(du0, u0, p, 0)
    f! = NonlinearFunction(local_rhs!;
     jac_prototype = float.(jac_sparsity))

    # Use NonlinearSolve to find steady state (f(u) = 0)
    prob = NonlinearProblem(f!, u0, p)
    
    # Solve using Newton-Raphson with Krylov linear solver (Matrix-free)
    # This is ideal for our stencil-based function.
    sol = solve(prob, NewtonRaphson(linsolve = LinearSolve.KrylovJL_GMRES()), 
                abstol=1e-6, reltol=1e-6)
    
    return sol.u
end

# ====================================================================
#  STEP 3: THE TESTSET
# ====================================================================

@testset "Olsthoorn V2 Verification" begin
    phi_ref, active_nodes, fxhd_nodes, inact_nodes = run_reference_solver()
    phi_gwflow = run_gwflow_solver_v2(KA.CPU())
    
    # Filter valid nodes
    valid_nodes = active_nodes .| fxhd_nodes
    phi_ref_valid = phi_ref[valid_nodes]
    phi_gw_valid = phi_gwflow[valid_nodes]
    
    # Check max difference
    diff = abs.(phi_gw_valid .- phi_ref_valid)
    max_diff = maximum(diff)
    @info "Max Difference: $max_diff"
    
    # Verify
    @test max_diff < 1e-2 # Slightly looser tol due to iterative solver
end
