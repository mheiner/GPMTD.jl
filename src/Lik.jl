# Lik.jl

export llik;

function ldensy_marg(y::Vector{T}, μ::T, σ2::T, Cov::PDMat) where T <: Real
    ## assumes mean zero
    n = length(y)
    Σ = Cov + σ2*I
    ldetΣ = logdet(Σ)
    d = (y - μ)
    Σinvd = Σ \ d

    -0.5 * ( n*log(2π) + ldetΣ + d'Σinvd )
end

## naive is faster than:
# logpdf( MvNormal( Cov + σ2*I ), y)

function ldensy_marg(y::Vector{T}, μ::T, σ2::T, κ::T, Cor::PDMat) where T <: Real
    ## assumes mean zero
    ## κ is the signal-to-noise ratio so that Σ = κ*σ2*Cor
    n = length(y)
    Σ0 = (κ .* Cor + I)
    ldetΣ = n*log(σ2) + logdet(Σ)
    d = (y - μ)
    Σinvd = Σ \ d

    -0.5 * ( n*log(2π) + ldetΣ + d'Σinvd )
end

function llik(model::Model_GPMTD)

    llik = 0.0
    lp = Vector{Float64}(undef, model.L+1)

    for i = 1:model.n

        lp[1] = model.state.lλ[1] - 0.5*log(2π*model.state.intercept.σ2) -
            0.5 * (model.y[i] - model.state.intercept.μ)^2 / model.state.intercept.σ2

        for j in 1:model.L
            lp[j+1] = model.state.lλ[j+1] -
            0.5*log(2π*model.state.mixcomps[j].σ2) -
                0.5 * (model.y[i] - model.state.mixcomps[j].μ - model.state.mixcomps[j].fx[i])^2 / model.state.mixcomps[j].σ2
        end

        # m = maximum(lp)
        # llik += m + log(sum(exp.(lp .- m))) # logsumexp
        llik += SparseProbVec.logsumexp(lp)
    end

    return llik
end
