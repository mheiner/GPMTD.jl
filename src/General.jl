# General.jl

export embed,
    NormalParams, ScInvChiSqParams,
    InterceptGPMTD, InterceptNormal,
    MixComponentGPMTD, MixComponentNormal,
    State_GPMTD,
    PriorIntercept, PriorIntercept_Normal,
    SNR_Hyper, SNR_Hyper_ScInvChiSq,
    PriorMixcomponent, PriorMixcomponent_Normal,
    PriorCovHyper, PriorCovHyper_Matern,
    Prior_GPMTD,
    Model_GPMTD;



function embed(y::Vector{T}, nlags::Int; out_sep::Bool=false) where T <: Number
    N = length(y)
    n = N - nlags
    emdim = nlags + 1
    XX = Array{T, 2}(undef, n, emdim)
    for i in 1:n
        tt = i + nlags
        XX[i,:] = copy(y[range(tt, step=-1, length=emdim)])
    end

    if out_sep
        out = (XX[:,1], XX[:,2:emdim])
    else
        out = XX
    end

    return out
end


mutable struct NormalParams
    μ::Real
    σ2::Real
    NormalParams(μ, σ2) = σ2 > 0.0 ? new(μ, σ2) : error("σ2 must be positive.")
end

mutable struct ScInvChiSqParams
    ν::Real
    s0::Real
    ScInvChiSqParams(ν, s0) = ν > 0.0 && s0 > 0.0 ? new(ν, s0) : error("Both parameters must be positive.")
end

mutable struct GammaParams
    shape::Real
    rate::Real
    GammaParams(shape, rate) = shape > 0.0 && rate > 0.0 ? new(shape, rate) : error("Both parameters must be positive.")
end

abstract type InterceptGPMTD end

mutable struct InterceptNormal <: InterceptGPMTD
    μ::Real
    σ2::Real
    InterceptNormal(μ, σ2) = σ2 > 0.0 ? new(μ, σ2) : error("σ2 must be positive.")
end
function InterceptNormal()
    InterceptNormal(0.0, 1.0)
end

abstract type SNR_Hyper end

mutable struct SNR_Hyper_ScInvChiSq <: SNR_Hyper
    κ_ν::Real
    κ0::Real
    SNR_Hyper_ScInvChiSq(κ_ν, κ0) = κ_ν > 0.0 && κ0 > 0.0 ? new(κ_ν, κ0) : error("Both parameters must be positive.")
end
function SNR_Hyper_ScInvChiSq()
    SNR_Hyper_ScInvChiSq(2.5, 3.5)
end

abstract type MixComponentGPMTD end

mutable struct MixComponentNormal <: MixComponentGPMTD
    μ::Real
    σ2::Real

    κ::Real # signal to noise ratio parameter
    κ_hypers::SNR_Hyper # this is updated elsewhere, but dragged along for each mixcomp.

    corParams::CorParams
    corHypers::CorHypers # this is updated elsewhere, but dragged along for each mixcomp.

    Cor::Matrix{<:Real}

    D::Matrix{<:Real} # full size
    fx::Vector{<:Real} # full size (all data points)

    ζon_indx::Vector{Int} # which y, x belong to this mixture component

    ## for MCMC
    rng::Union{MersenneTwister, Threefry4x}
    cSig::PDMat
    runningsum_Met::Union{Array{<:Real, 1}, Nothing}
    runningSS_Met::Union{Array{<:Real, 2}, Nothing}
    accpt::Int
    adapt::Bool
    adapt_iter::Int
end
function MixComponentNormal()

    Cor = Matrix(LinearAlgebra.Diagonal([1.0]))
    D = zeros(Float64, 1, 1)
    fx = zeros(Float64, 1)
    ζon_indx = ones(Int, 1)

    rng = MersenneTwister(0)
    cSig = PDMat(Matrix(LinearAlgebra.Diagonal([1.0])))

    return MixComponentNormal(0.0, 1.0, 1.0, SNR_Hyper_ScInvChiSq(), MaternParams(), MaternHyper_ScInvChiSq(),
        Cor, D, fx, ζon_indx, rng, cSig, nothing, nothing, 0, false, 0)
end


mutable struct State_GPMTD

    ## parameters
    intercept::InterceptGPMTD
    mixcomps::Vector{<:MixComponentGPMTD}

    lλ::Vector{<:Real}
    ζ::Vector{Int}

    ## others
    iter::Int
    accpt::Vector{Int}

    adapt::Bool
    adapt_iter::Union{Int, Nothing}

    llik::Real

end

## outer constructor
function State_GPMTD(X::Matrix{T}, intercept::InterceptNormal,
    mixcomps::MixComponentNormal) where T <: Real

    n, R = size(X)

    mixcomps_out = [ deepcopy(mixcomps) for ℓ = 1:R ]
    mixcomps_out[1].rng = Random123.Threefry4x( tuple(rand(Int, 4)...) ) # parallel-safe generator

    ## populate mixcomps
    for ℓ = 1:R
        mixcomps_out[ℓ].D = pairDistMat(X[:,[ℓ]])
        mixcomps_out[ℓ].Cor = covMat(mixcomps_out[ℓ].D, mixcomps_out[ℓ].κ*mixcomps_out[ℓ].σ2, mixcomps_out[ℓ].corParams)
        mixcomps_out[ℓ].fx = fill(1.0, n)
        mixcomps_out[ℓ].ζon_indx = Vector{Int}(undef, 0)

        mixcomps_out[ℓ].rng = deepcopy(mixcomps_out[1].rng)
        Random123.set_counter!(mixcomps_out[ℓ].rng, ℓ+1)

        mixcomps_out[ℓ].cSig = PDMat(Matrix(LinearAlgebra.Diagonal(fill(0.1, 2))))
        mixcomps_out[ℓ].runningsum_Met = zeros(Float64, 2)
        mixcomps_out[ℓ].runningSS_Met = zeros(Float64, 2, 2)
    end

    lλ = fill( log(1.0/(R+1.0)), R + 1 )
    ζ = zeros(Int, n)

    return State_GPMTD(intercept, mixcomps_out, lλ, ζ, 0, zeros(Int, R), false, 0, 0.0)
end
## if you want to do another constructor with random inits, create another function with the prior


abstract type PriorIntercept end

mutable struct PriorIntercept_Normal <: PriorIntercept
    μ::NormalParams
    σ2::ScInvChiSqParams
end
function PriorIntercept_Normal()
    μ_μ = 0.0
    μ_σ2 = 1.0e3
    prior_μ = NormalParams(μ_μ, μ_σ2)

    σ2_ν = 5.0
    σ2_s0 = 1.0
    prior_σ2 = ScInvChiSqParams(σ2_ν, σ2_s0)

    return PriorIntercept_Normal( prior_μ, prior_σ2 )
end

abstract type PriorMixcomponent end

mutable struct PriorMixcomponent_Normal <: PriorMixcomponent
    μ::NormalParams
    σ2::ScInvChiSqParams
end
function PriorMixcomponent_Normal()

    μ_μ = 0.0
    μ_σ2 = 1.0e3
    prior_μ = NormalParams(μ_μ, μ_σ2)

    σ2_ν = 5.0
    σ2_s0 = 1.0
    prior_σ2 = ScInvChiSqParams(σ2_ν, σ2_s0)

    return PriorMixcomponent_Normal(prior_μ, prior_σ2)
end

abstract type PriorCovHyper end

mutable struct PriorCovHyper_Matern <: PriorCovHyper
    κ_ν::Vector{<:Real}
    κ0::GammaParams

    lenscale_ν::Vector{<:Real}
    lenscale0::GammaParams
end
function PriorCovHyper_Matern()
    κ_ν = [2.5, 3.5, 5.0, 7.5, 10.0]
    κ0 = GammaParams(3.5*2.0, 2.0)

    lenscale_ν = [2.5, 3.5, 5.0, 7.5, 10.0]
    lenscale0 = GammaParams(1.5*2.0, 2.0)

    return PriorCovHyper_Matern(κ_ν, κ0, lenscale_ν, lenscale0)
end

mutable struct Prior_GPMTD
    intercept::PriorIntercept
    mixcomps::Vector{<:PriorMixcomponent}
    covhyper::PriorCovHyper
    λ::Union{SparseDirMix, SBMprior, Vector{Float64}}
end

## Default prior constructor
function Prior_GPMTD(R::Int, intercept::InterceptNormal,
    mixcomps::MixComponentNormal)

    intcpt_prior = PriorIntercept_Normal()
    mixcomps_prior = [ PriorMixcomponent_Normal() for ℓ = 1:R ]
    covhyper_prior = PriorCovHyper_Matern()
    λ_prior = SparseProbVec.SBMprior(R+1, 1.0e3, 0.5, 0.25, 1.0, 1.0)

    Prior_GPMTD(intcpt_prior, mixcomps_prior, covhyper_prior, λ_prior)
end

mutable struct Model_GPMTD

    ## Provided
    y::Vector{<:Real}
    X::Matrix{<:Real}
    prior::Prior_GPMTD
    state::State_GPMTD

    ## Calculated
    n::Int # length of data
    R::Int # number of lags considered
    D::Vector{Matrix{<:Real}} # Distance matrix for each column of X

end

## outer constructor
function Model_GPMTD(y::Vector{T}, X::Matrix{T}, prior::Prior_GPMTD,
    state::State_GPMTD) where T <: Real

    n = length(y)
    nx, R = size(X)

    n == nx || throw("Lengths of X and y differ.")

    D = [ pairDistMat(X[:,[ℓ]]) for ℓ = 1:R ]

    return Model_GPMTD(y, X, prior, state, n, R, D)
end
