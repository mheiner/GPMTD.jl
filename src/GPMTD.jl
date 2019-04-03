module GPMTD

## distributed packages
using Distances
using Distributions
using PDMats
using Random
using LinearAlgebra
import StatsBase: counts #, mean

## personal packages
using BayesInference
using SparseProbVec
# using MTD

## files
include("General.jl")
include("Dist.jl")
include("Cor.jl")
include("Lik.jl")
include("Update_int.jl")
include("Update_F.jl")
include("Update_Cov.jl")
include("Update_alloc.jl")
include("Update_weights.jl")

end # module
