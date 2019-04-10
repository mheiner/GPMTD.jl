module GPMTD

## distributed packages
using Distances
using Distributions
using PDMats
using Random, Random123
using LinearAlgebra
using Distributed
import StatsBase: counts, sample, Weights

## personal packages
using SparseProbVec

## files
include("General.jl")
include("Dist.jl")
include("Cor.jl")
include("Lik.jl")
include("Update_int.jl")
include("Update_Cov.jl")
include("Update_mixcomps.jl")
include("Update_alloc.jl")
include("Update_weights.jl")
include("Sim.jl")

end # module
