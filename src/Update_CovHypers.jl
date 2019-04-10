# Update_CovHypers.jl

function llik_ScInvChiSq(y::Vector{T}, ν::T, s0::T) where T <: Real
    sum1divy = sum(1.0 ./ y)
    sumlogy = sum(log.(y))
    n = length(y)

    halfnu = 0.5 * ν
    a1 = halfnu * n * (log(ν) + log(s0) - log(2.0))
    a2 = n * lgamma(halfnu)
    a3 = halfnu * sumlogy
    a4 = halfnu * s0 * sum1divy

    return a1 - a2 - a3 - a4
end

function rpost_ScInvChiSq_s0(y::Vector{T}, ν::T, prior::GammaParams)
    sum1divy = sum(1.0 ./ y)
    n = length(y)

    halfnu = 0.5 * ν
    a1 = prior.shape + halfnu * n
    b1 = prior.rate + halfnu * sum1divy

    return rand(Distributions.Gamma(a1, 1.0/b1))
end

function update_cov_hypers!(state::State_GPMTD, prior::PriorCovHyper_Matern) where T <: Real

    R = length(state.mixcomps)

    nζ = StatsBase.counts(state.ζ, 1:R) # intercept does not contain these parameters
    use_indx = findall( nζ .> 0 )

    if length(use_indx) > 0

        κ_dat = [ state.mixcomp[j].κ for j in use_indx ]
        lenscale_dat = [ state.mixcomp[j].corParams.lenscale for j in use_indx ]



    else


    end

    return nothing
end
