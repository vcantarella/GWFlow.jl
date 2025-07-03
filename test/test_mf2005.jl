# create a regular grid and solve a simple groundwater flow problem
using GWSolve
using Test

# create a regular grid
dx = 10
dy = 10
dz = 1
xl = 0
yu = 1000
zt = 50
nx = 100
ny = 100
nz = 50
grid = GWSolve.regular_grid(xl, yu, zt, dx, dy, dz, nx, ny, nz)
# create a constant head boundary condition
ind = [(lay, row, 1) for lay in 1:nz, row in 1:ny]
# flatten ind
ind_flat = reshape(ind, length(ind))
head = fill(1000.0, length(ind_flat))
cbc = GWSolve.ConstantHead(ind_flat, head)
ind = [(lay, row, nx) for lay in 1:nz, row in 1:ny]
ind_flat = reshape(ind, length(ind))
head = fill(200, length(ind_flat))
cbc2 = GWSolve.ConstantHead(ind_flat, head)
x_edges = grid.x_e
y_edges = grid.y_e
z_edges = grid.z_e
