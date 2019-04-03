# Update_weights.jl

function update_lλ(α0_λ::Vector{Float64}, ζ::Vector{Int}, R::Int)

    Nζ = StatsBase.counts(ζ, 0:R)
    α1_λ = α0_λ .+ Nζ
    lλ_out = SparseProbVec.rDirichlet(α1_λ, logout=true)

    lλ_out
end
function update_lλ(prior_λ::SparseDirMix, ζ::Vector{Int}, R::Int)

    Nζ = StatsBase.counts(ζ, 0:R)
    α1_λ = prior_λ.α .+ Nζ
    d = SparseProbVec.SparseDirMix(α1_λ, prior_λ.β)
    lλ_out = SparseProbVec.rand(d, logout=true)

    lλ_out
end
function update_lλ(prior_λ::SBMprior, ζ::Vector{Int}, R::Int)

    Nζ = StatsBase.counts(ζ, 0:R)
    post_lλ = SparseProbVec.SBM_multinom_post(prior_λ, Nζ)
    lλ_out = SparseProbVec.rand(post_lλ, logout=true)

    lλ_out
end
