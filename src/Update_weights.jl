# Update_weights.jl

function update_lλ!(state::State_GPMTD, prior_λ::Vector{Float64})

    L = length(state.lλ) - 1

    Nζ = StatsBase.counts(state.ζ, 0:L)
    α1_λ = prior_λ .+ Nζ
    state.lλ = SparseProbVec.rDirichlet(α1_λ, logout=true)

    return nothing
end
function update_lλ!(state::State_GPMTD, prior_λ::SparseDirMix)

    L = length(state.lλ) - 1

    Nζ = StatsBase.counts(state.ζ, 0:L)
    α1_λ = prior_λ.α .+ Nζ
    d = SparseProbVec.SparseDirMix(α1_λ, prior_λ.β)
    state.lλ = SparseProbVec.rand(d, logout=true)

    return nothing
end
function update_lλ!(state::State_GPMTD, prior_λ::SBMprior)

    L = length(state.lλ) - 1

    Nζ = StatsBase.counts(state.ζ, 0:L)
    post_lλ = SparseProbVec.SBM_multinom_post(prior_λ, Nζ)
    state.lλ = SparseProbVec.rand(post_lλ, logout=true)

    return nothing
end
