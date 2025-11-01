# This file is part of your GroundwaterFlow.jl package
using KernelAbstractions
using GWGrids

"""
    calculate_cell_flows_kernel!(frf, fff, flf, head_3d, Cx, Cy, Cz, grid)

A KernelAbstractions kernel to compute cell-by-cell flows from solved heads
and pre-computed conductances.
"""
@kernel function calculate_cell_flows_kernel!(
    frf, fff, flf, # Output arrays
    head_3d,       # Input: solved heads
    Cx, Cy, Cz,    # Input: conductances
    grid           # Input: grid info
)
    l, r, c = @index(Global, NTuple)
    
    nlay, nrow, ncol = grid.nlay, grid.nrow, grid.ncol
    
    # Get the head at the current cell
    h1 = head_3d[l, r, c]
    
    # --- Flow Right Face (X-direction) ---
    if c < ncol
        h2 = head_3d[l, r, c+1]
        frf[l, r, c] = Cx[l, r, c] * (h1 - h2)
    end
    
    # --- Flow Front Face (Y-direction) ---
    # Note: MODFLOW 'front' is from row r to r+1
    if r < nrow
        h2 = head_3d[l, r+1, c]
        fff[l, r, c] = Cy[l, r, c] * (h1 - h2)
    end
    
    # --- Flow Lower Face (Z-direction) ---
    # Note: MODFLOW 'lower' is from layer l to l+1
    if l < nlay
        h2 = head_3d[l+1, r, c]
        flf[l, r, c] = Cz[l, r, c] * (h1 - h2)
    end
end

"""
    calculate_flows(head_solution, Cx, Cy, Cz, grid)

High-level wrapper to orchestrate the cell-by-cell flow calculation.
"""
function calculate_flows(
    head_3d::AbstractArray{T, 3}, 
    Cx::AbstractArray{T, 3}, 
    Cy::AbstractArray{T, 3}, 
    Cz::AbstractArray{T, 3}, 
    grid::PlanarRegularGrid
) where T
    
    backend = KernelAbstractions.get_backend(head_3d)
    nlay, nrow, ncol = grid.nlay, grid.nrow, grid.ncol
    
    # 1. Allocate output arrays on the same device as the head solution
    frf = similar(head_3d); frf .= zero(T)
    fff = similar(head_3d); fff .= zero(T)
    flf = similar(head_3d); flf .= zero(T)
    
    # 2. Launch the kernel
    kernel = calculate_cell_flows_kernel!(backend)
    kernel(
        frf, fff, flf, 
        head_3d, 
        Cx, Cy, Cz, 
        grid; 
        ndrange=(nlay, nrow, ncol)
    )
    
    # 3. Wait for the kernel to finish
    synchronize(backend)
    
    # 4. Return a NamedTuple of the results
    return (right=frf, front=fff, lower=flf)
end
