module GPMTD

## distributed packages
using Distances
using Distributions
using PDMats
using Random, Random123
using LinearAlgebra
using Distributed
using Dates
# import StatsBase: counts, sample, Weights
using StatsBase
using SpecialFunctions

## personal packages
using SparseProbVec

## files
include("Cor.jl")
include("General.jl")
include("Dist.jl")
include("Update_int.jl")
include("Update_Cov.jl")
include("Update_mixcomps.jl")
include("Update_alloc.jl")
include("Update_weights.jl")
include("Update_CovHypers.jl")
include("Lik.jl")
include("Sim.jl")
include("mcmc.jl")
include("DensityEstimation.jl")

end # module
