abstract type BC
end
struct ConstantHead <: BC
    index::AbstractArray{NTuple{3, <:Int}, 1}
    head::AbstractArray{<:Real, 1}
end
struct ConstantFlux <: BC
    index::AbstractArray{NTuple{3, <:Int}, 1}
    spec_disch::AbstractArray{<:Real, 1}
    face::AbstractArray{<:Int, 1}  # MODPATH convention
end
struct NoFlow <: BC
    index::AbstractArray{NTuple{3, <:Int}, 1}
    face::AbstractArray{<:Int, 1}  # MODPATH convention
end
struct GeneralHead <: BC
    index::AbstractArray{NTuple{3, <:Int}, 1}
    cond::AbstractArray{<:Real, 3}
    head::AbstractArray{<:Real, 3}
    face::AbstractArray{<:Int, 1}  # MODPATH convention
end
function apply_bound!(du, u, k, bc::BC)
end
function apply_bound!(du, u, k, bc::ConstantHead)
    for i in eachindex(bc.index)
        u[bc.index[i]] = bc.head[i]
    end
end
function apply_bound!(du, u, k, dx, dy, dz, ibound,bc::ConstantFlux)
    for i in eachindex(bc.index)
        if face[i] == 1 | face[i] == 2
            if ibound == 1
                du[bc.index[i]] += bc.spec_disch[i] * dy * dz  # Assuming dy and dz are defined
            elseif ibound == 2
                if z + dz > u[bc.index[i]]
                    du[bc.index[i]] += bc.spec_disch[i] * dy * dz  # Assuming dy and dz are defined
                else
                    du[bc.index[i]] += bc.spec_disch[i] * dy * u[bc.index[i]] # Assuming dy and dz are defined
                end
            end
        elseif face[i] == 3 | face[i] == 4
            if ibound == 1
                du[bc.index[i]] += bc.spec_disch[i] * dx * dz  # Assuming dx and dz are defined
            elseif ibound == 2
                if z + dz > u[bc.index[i]]
                    du[bc.index[i]] += bc.spec_disch[i] * dx * dz  # Assuming dx and dz are defined
                else
                    du[bc.index[i]] += bc.spec_disch[i] * dx * u[bc.index[i]] # Assuming dx and dz are defined
                end
            end
        elseif face[i] == 5 | face[i] == 6
            du[bc.index[i]] += bc.spec_disch[i] * dx * dy  # Assuming dx and dy are defined
        end
    end
end
function apply_bound!(du, u, k, dx, dy, dz, ibound,bc::GeneralHead)
    for i in eachindex(bc.index)
        if face[i] == 1 | face[i] == 2
            if ibound == 1
                du[bc.index[i]] += bc.cond[i] * dy * dz / dx  # Assuming dy and dz are defined
            elseif ibound == 2
                if z + dz > u[bc.index[i]]
                    du[bc.index[i]] += bc.cond[i] * dy * dz / dx  # Assuming dy and dz are defined
                else
                    du[bc.index[i]] += bc.cond[i] * dy * u[bc.index[i]] / dx # Assuming dy and dz are defined
                end
            end
        elseif face[i] == 3 | face[i] == 4
            if ibound == 1
                du[bc.index[i]] += bc.cond[i] * dx * dz / dy  # Assuming dx and dz are defined
            elseif ibound == 2
                if z + dz > u[bc.index[i]]
                    du[bc.index[i]] += bc.cond[i] * dx * dz / dy  # Assuming dx and dz are defined
                else
                    du[bc.index[i]] += bc.cond[i] * dx * u[bc.index[i]] / dy # Assuming dx and dz are defined
                end
            end
        elseif face[i] == 5 | face[i] == 6
            du[bc.index[i]] += bc.cond[i] * dx * dy / dz # Assuming dx and dy are defined
        end
    end
end

