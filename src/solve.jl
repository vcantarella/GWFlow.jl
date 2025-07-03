include("grids.jl")

function harmonic_mean(k1, k2, dx1, dx2)
    # Calculate the harmonic mean of two hydraulic conductivities
    return 1 / (dx1/(2*k1) + dx2/(2*k2))
end


function build_regular_rhs(grid, dx, dy, dz, ibound, k::AbstractArray{<:Number, 3}, bcs:: AbstractArray{<:BC},
    apply_p_to_model!::Function)
    # Build the right-hand side of the equation based on the grid, hydraulic conductivity, and boundary conditions
    # This is a placeholder implementation; actual implementation will depend on the specific problem
    let dx = dx, dy = dy, dz = dz,
        k = k, bcs = bcs, ibound = ibound, grid = grid,
        laycon = grid.laycon,
        # we dont need to loop over every cell, but to loop at every edge!
        # there are a few situations where we need to modify just looping at the edges:
        # 1. when inactive cells: ignore the inactive cell edges.
        # 2. When a boundary cell is of type general head. Then the edge needs to be enhanced.
        xedges = grid.x_e,
        yedges = grid.y_e,
        zedges = grid.z_e

        function rhs(du, u, p, t)
            # we take the connections of the grid. These are the loops we need to do.
            # 1. loop over the edges in the x direction
            apply_p_to_model!(k, bcs, u, p, t)
            for edge in xedges[2:end-1]
                for row in axes(u, 2), lay in axes(u, 1)
                    if ibound[lay, row, edge] == 0  | ibound[lay, row, edge-1] == 0 # inactive cell
                        continue
                    end
                    if laycon[lay] > 0 # unconfined layer
                        if u[lay, row, edge-1] - zedges[lay+1] > dz
                            delz1 = dz
                        else
                            delz1 = u[lay, row, edge-1] - zedges[lay+1]
                        end
                        if u[lay, row, edge] - zedges[lay+1] > dz
                            delz2 = dz
                        else
                            delz2 = u[lay, row, edge] - zedges[lay+1]
                        end
                        a1 = dy * delz1
                        a2 = dy * delz2
                        c1 = k[lay, row, edge-1] * a1 / dx
                        c2 = k[lay, row, edge] * a2 / dx
                        C = c1*c2 / (c1 + c2)
                    else
                        a1 = a2 = dy * dz
                        c1 = k[lay, row, edge-1] * a1 / dx
                        c2 = k[lay, row, edge] * a2 / dx
                        C = c1*c2 / (c1 + c2)
                    end
                    du[lay, row, edge] += C * (u[lay, row, edge] - u[lay, row, edge-1]) / dx
                    du[lay, row, edge-1] -= C * (u[lay, row, edge] - u[lay, row, edge-1]) / dx
                end
            end
            for edge in yedges[2:end-1]
                for col in axes(u, 3), lay in axes(u, 1)
                    if ibound[lay, edge, col] == 0 | ibound[lay, edge-1, col] == 0 # inactive cell
                        continue
                    end
                    if laycon[lay] > 0 # unconfined layer
                        if u[lay, edge-1, col] - zedges[lay+1] > dz
                            delz1 = dz
                        else
                            delz1 = u[lay, edge-1, col] - zedges[lay+1]
                        end
                        if u[lay, edge, col] - zedges[lay+1] > dz
                            delz2 = dz
                        else
                            delz2 = u[lay, edge, col] - zedges[lay+1]
                        end
                        a1 = dx * delz1
                        a2 = dx * delz2
                        c1 = k[lay, edge-1, col] * a1 / dy
                        c2 = k[lay, edge, col] * a2 / dy
                        C = c1*c2 / (c1 + c2)
                    else
                        a1 = a2 = dx * dz
                        c1 = k[lay, edge-1, col] * a1 / dy
                        c2 = k[lay, edge, col] * a2 / dy
                        C = c1*c2 / (c1 + c2)
                    end
                    du[lay, edge, col] += C * (u[lay, edge, col] - u[lay, edge-1, col]) / dy
                    du[lay, edge-1, col] -= C * (u[lay, edge, col] - u[lay, edge-1, col]) / dy
                end
            end
            for edge in zedges[2:end-1]
                for col in axes(u, 3), row in axes(u, 2)
                    if ibound[edge, row, col] == 0 | ibound[edge-1, row, col] == 0 # inactive cell
                        continue
                    end
                    a1 = a2 = dx * dy
                    c1 = k[edge-1, row, col] * a1 / dz
                    c2 = k[edge, row, col] * a2 / dz
                    C = c1*c2 / (c1 + c2)
                    du[edge, row, col] += C * (u[edge, row, col] - u[edge-1, row, col]) / dz
                    du[edge-1, row, col] -= C * (u[edge, row, col] - u[edge-1, row, col]) / dz
                end
            end
            for bc in bcs
                apply_bound!(du, u, k, dx, dy, dz, ibound, bc)
            end
        end
    end
    return rhs
end