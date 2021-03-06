# Dist.jl

export pairDistMat, expanD;

## These functions require matrices; the output will have extent equal to the number of rows of X.
function pairDistMat(X::Matrix{T}, metric=Euclidean()) where T <: Real
    Distances.pairwise(metric, Matrix(X'), dims=2)
end

function pairDistMat(X::Matrix{T}, Y::Matrix{T},
    metric=Euclidean()) where T <: Real
    Distances.pairwise(metric, Matrix(X'), Matrix(Y'), dims=2)
end

function expanD(D::Matrix{T}, x_new::Union{T, Vector{T}},
    x_old::Matrix{T}; front::Bool=true) where T <: Real
    ## Note that x_new is one new point only

    if typeof(x_new) <: Real
        x_new = [x_new]
    end

    x_new = reshape(x_new, 1, length(x_new))
    d_new = vec(pairDistMat(x_new, x_old))

    if front
        A = vcat(0.0, d_new)
        B = Matrix( hcat( d_new, D )' )

        Out = hcat( A, B )
    else
        A = Matrix( hcat( D, d_new ) )
        B = vcat(d_new, 0.0)

        Out = vcat( A, B' )
    end

    return Out
end
