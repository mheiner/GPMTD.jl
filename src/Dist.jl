# Dist.jl

export pairDistMat;

function pairDistMat(X::Union{Array{T,1}, Array{T,2}}, metric=Euclidean()) where T <: Real
    Distances.pairwise(metric, Matrix(X'), dims=2)
end

function pairDistMat(X::Union{Array{T,1}, Array{T,2}}, Y::Union{Array{T,1}, Array{T,2}},
    metric=Euclidean()) where T <: Real
    Distances.pairwise(metric, Matrix(X'), Matrix(Y'), dims=2)
end
