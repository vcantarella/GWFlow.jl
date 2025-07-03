# Trying to reproduce the Gwflow model from Prof. Olsthoorn's lecture notes
# https://olsthoorn.readthedocs.io/en/latest/02_fin_dif_modeling.html
# Using just normal Julia code, not the Gwflow package

using LinearAlgebra
using SparseArrays
using CairoMakie

# Specify a rectangular grid
x = -1000.0:25.0:1000.0
y = -1000.0:25.0:1000.0  # Note: Julia ranges are naturally forward
z = -100.0:20.0:0.0

# Get number of cells along each axis
Nx = length(x) - 1
Ny = length(y) - 1  
Nz = length(z) - 1

sz = (Nz, Ny, Nx)  # Shape of the model
Nod = prod(sz)     # Total number of cells

# Cell dimensions
dx = reshape(diff(x), 1, 1, Nx)
dy = reshape(diff(y), 1, Ny, 1)  
dz = reshape(abs.(diff(z)), Nz, 1, 1)

# IBOUND array - boundary conditions
IBOUND = ones(Int, sz)
IBOUND[:, end, :] .= -1     # Last row has prescribed heads
IBOUND[:, 41:45, 21:70] .= 0  # Inactive cells (Julia is 1-indexed)

# Boolean arrays for active, inactive, and fixed head cells
active = reshape(IBOUND .> 0, Nod)
inact = reshape(IBOUND .== 0, Nod)
fxhd = reshape(IBOUND .< 0, Nod)

# Hydraulic conductivities
k = 10.0  # m/d uniform conductivity
kx = k * ones(sz)
ky = k * ones(sz) 
kz = k * ones(sz)

# Half cell flow resistances
Rx = 0.5 * dx ./ (dy .* dz) ./ kx
Ry = 0.5 * dy ./ (dz .* dx) ./ ky
Rz = 0.5 * dz ./ (dx .* dy) ./ kz

# Make inactive cells inactive by setting resistance to Inf
Rx_flat = reshape(Rx, Nod)
Rx_flat[inact] .= Inf
Rx = reshape(Rx_flat, sz)

Ry_flat = reshape(Ry, Nod)
Ry_flat[inact] .= Inf
Ry = reshape(Ry_flat, sz)

Rz_flat = reshape(Rz, Nod)
Rz_flat[inact] .= Inf
Rz = reshape(Rz_flat, sz)

# Conductances between adjacent cells
Cx = 1 ./ (Rx[:, :, 1:end-1] .+ Rx[:, :, 2:end])
Cy = 1 ./ (Ry[:, 1:end-1, :] .+ Ry[:, 2:end, :])
Cz = 1 ./ (Rz[1:end-1, :, :] .+ Rz[2:end, :, :])

# Node numbering array
NOD = reshape(0:Nod-1, sz)  # Julia is 0-indexed for this purpose

# Neighbor identification
IE = NOD[:, :, 2:end]      # Eastern neighbors
IW = NOD[:, :, 1:end-1]    # Western neighbors  
IN = NOD[:, 1:end-1, :]    # Northern neighbors
IS = NOD[:, 2:end, :]      # Southern neighbors
IT = NOD[1:end-1, :, :]    # Top neighbors
IB = NOD[2:end, :, :]      # Bottom neighbors

# Build sparse system matrix
# Convert to 1-based indexing for Julia
row_indices = vcat(vec(IE), vec(IW), vec(IN), vec(IS), vec(IB), vec(IT)) .+ 1
col_indices = vcat(vec(IW), vec(IE), vec(IS), vec(IN), vec(IT), vec(IB)) .+ 1
values = vcat(vec(Cx), vec(Cx), vec(Cy), vec(Cy), vec(Cz), vec(Cz))

A = sparse(row_indices, col_indices, values, Nod, Nod)

# Diagonal elements (negative sum of row coefficients)
adiag = -vec(sum(A, dims=2))
A_diag = sparse(1:Nod, 1:Nod, adiag, Nod, Nod)

# Complete system matrix
A_complete = A + A_diag

# Fixed flows boundary conditions
FQ = zeros(sz)
FQ[3, 31, 26] = -1200.0  # Extraction in this cell (1-indexed)

# Initial heads
HI = zeros(sz)

# Right-hand side vector
RHS = vec(FQ) - A_complete[:, fxhd] * vec(HI)[fxhd]

# Solve for unknown heads (active cells only)
Phi = vec(HI)
Phi[active] = A_complete[active, active] \ RHS[active]

# Set inactive cells to NaN
Phi[inact] .= NaN

# Reshape to grid dimensions
Phi = reshape(Phi, sz)

# Plotting
xm = 0.5 * (x[1:end-1] + x[2:end])  # Cell centers
ym = 0.5 * (y[1:end-1] + y[2:end])
layer = 3  # Layer to plot (1-indexed)
nc = 50    # Number of contours

# Create contour plot with CairoMakie
fig = Figure(size = (800, 600))
ax = Axis(fig[1, 1], 
    aspect = DataAspect(),
    xlabel = "x [m]", 
    ylabel = "y [m]",
    title = "Contours ($nc in total) of the head in layer $layer with inactive section")

contour!(ax, xm, ym, Phi[layer, :, :], levels = nc)

fig

