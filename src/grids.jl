struct RegularGrid{C, T}
    x_e::Vector{C}
    y_e::Vector{C}
    z_e::Vector{C}
    x_c::Vector{C}
    y_c::Vector{C}
    z_c::Vector{C}
    u::Array{T, 3}  # Assuming u is a 3D array
end

function regular_grid(xl, yu, zt, dx, dy, dz, nx, ny, nz)
    # Create a regular grid with specified dimensions and spacing
    x_edges = xl:dx:xl + dx * nx
    y_edges = yu:-dy:yu - dy * ny
    z_edges = zt:-dz:zt - dz * nz

    x_centers = x_edges[1:end-1] .+ dx / 2
    y_centers = y_edges[1:end-1] .+ dy / 2
    z_centers = z_edges[1:end-1] .+ dz / 2

    u = zeros(length(z_centers), length(y_centers), length(x_centers))
    grid = RegularGrid{Float64, Float64}(
        x_edges,
        y_edges,
        z_edges,
        x_centers,
        y_centers,
        z_centers,
        u,
    )
    return grid
end