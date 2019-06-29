# Update_CovHypers.jl

function llik_ScInvChiSq(y::Vector{T}, ν::T, s0::T) where T <: Real
    sum1divy = sum(1.0 ./ y)
    sumlogy = sum(log.(y))
    n = length(y)

    halfnu = 0.5 * ν
    a1 = halfnu * n * (log(ν) + log(s0) - log(2.0))
    a2 = n * SpecialFunctions.lgamma(halfnu)
    a3 = halfnu * sumlogy
    a4 = halfnu * s0 * sum1divy

    return a1 - a2 - a3 - a4
end

function rpost_ScInvChiSq_s0(y::Vector{T}, ν::T, prior::GammaParams) where T <: Real
    sum1divy = sum(1.0 ./ y)
    n = length(y)

    halfnu = 0.5 * ν
    a1 = prior.shape + halfnu * n
    b1 = prior.rate + halfnu * sum1divy

    return rand(Distributions.Gamma(a1, 1.0/b1))
end

function rpost_ScInvChiSq_ν(y::Vector{T}, s0::T, prior::Vector{T}) where T <: Real
    ## Here the prior is a vector of values representing a discrete uniform over them.
    lp = [ llik_ScInvChiSq(y, nu, s0) for nu in prior ]
    w = exp.( lp .- maximum(lp) )
    return StatsBase.sample(prior, Weights(w))
end

function update_cov_hypers!(state::State_GPMTD, prior::PriorCovHyper_Matern) where T <: Real

    L = length(state.mixcomps)

    nζ = StatsBase.counts(state.ζ, 1:L) # intercept does not involve these parameters
    use_indx = findall( nζ .> 0 )

    if length(use_indx) > 0

        κ_dat = [ state.mixcomps[j].κ for j in use_indx ]
        lenscale_dat = [ state.mixcomps[j].corParams.lenscale for j in use_indx ]

        κ_ν_out = rpost_ScInvChiSq_ν(κ_dat, state.mixcomps[1].κ_hypers.κ0, prior.κ_ν)
        κ0_out = rpost_ScInvChiSq_s0(κ_dat, κ_ν_out, prior.κ0)

        lenscale_ν_out = rpost_ScInvChiSq_ν(lenscale_dat, state.mixcomps[1].corHypers.lenscale0, prior.lenscale_ν)
        lenscale0_out = rpost_ScInvChiSq_s0(lenscale_dat, lenscale_ν_out, prior.lenscale0)

    else

        κ_ν_out = StatsBase.sample(prior.κ_ν)
        κ0_out = rand(Distributions.Gamma(prior.κ0.shape, 1.0/prior.κ0.rate))

        lenscale_ν_out = StatsBase.sample(prior.lenscale_ν)
        lenscale0_out = rand(Distributions.Gamma(prior.lenscale0.shape, 1.0/prior.lenscale0.rate))

    end

    # update these parameters in all mixcomps
    for ℓ = 1:L
        state.mixcomps[ℓ].κ_hypers.κ_ν = κ_ν_out
        state.mixcomps[ℓ].κ_hypers.κ0 = κ0_out
        state.mixcomps[ℓ].corHypers.lenscale_ν = lenscale_ν_out
        state.mixcomps[ℓ].corHypers.lenscale0 = lenscale0_out
    end

    return nothing
end



### tests passed 4/12/19
# ν = 7.5
# s0 = 10.0
# using Distributions
# using StatsBase
# using SpecialFunctions
# n = 1000
#
# xx = rand(InverseGamma(0.5*ν, 0.5*ν*s0), n)
#
# nsamp = 1000
# samp = ones(nsamp, 2)
# for i = 2:nsamp
#     samp[i,1] = rpost_ScInvChiSq_ν(xx, samp[i-1,2], [5.0, 7.5, 10.0, 25.0, 50.0])
#     samp[i,2] = rpost_ScInvChiSq_s0(xx, samp[i,1], GammaParams(1.0, 1.0))
# end
#
# using Plotly
# plot(samp[:,1])
# plot(samp[:,2])
