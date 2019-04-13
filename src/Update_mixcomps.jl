# Update_mixcomps.jl

function update_μ_σ2!(mixcomp::MixComponentNormal, suffstat::Dict{Symbol, T},
    prior::PriorMixcomponent_Normal) where T <: Real

    ## mean
    v1 = 1.0 / ( 1.0 / prior.μ.σ2 + suffstat[:sumWinv] / mixcomp.σ2 )
    sd1 = sqrt(v1)
    μ1 = v1 * ( prior.μ.μ / prior.μ.σ2 + suffstat[:sumWinv] * suffstat[:yhat] / mixcomp.σ2 )

    mixcomp.μ = rand(mixcomp.rng, Distributions.Normal(μ1, sd1) )

    ## variance
    a1 = 0.5 * ( prior.σ2.ν + length(mixcomp.ζon_indx) )
    ss = suffstat[:sumWinv]*(suffstat[:yhat] - mixcomp.μ)^2 + suffstat[:s2]
    b1 = 0.5 * ( prior.σ2.ν * prior.σ2.s0 + ss )

    mixcomp.σ2 = rand(mixcomp.rng, Distributions.InverseGamma(a1, b1) )

    return nothing
end

function rpost_f(y::Vector{T}, μ::T, σ2::T, κ::T, Cor::Symmetric{Float64, Matrix{Float64}},
    rng::Union{MersenneTwister, Threefry4x}) where T <: Real
    # conjugate update for MvNormal mean with prior mean μ, known general
    #   correlation matrix Cor, signal-to-noise ratio κ, and indep. noisy obs with variance σ2.

    Corinv = inv(Cor)
    J = PDMat_adj((Corinv ./ κ + I) ./ σ2)

    h = ( y .- μ ) ./ σ2

    rand(rng, Distributions.MvNormalCanon(h, J) )
end

function rfullcond_fstar(f_ell::T, Celel::T, Cstst::T, Cstel::T,
    rng::Union{MersenneTwister, Threefry4x}, tol=1.0e-10) where T <: Real

    ## here C is the **Covariance** matrix

    μ = Cstel * ( Celel \ f_ell )
    Σ = Cstst - Cstel * ( Celel \ Cstel )

    if Σ < 0.0 && abs(Σ) < tol
        Σ = abs(Σ)
    end

    return sqrt(Σ) * randn(rng) + μ
end
function rfullcond_fstar(f_ell::T, Celel::T,
    Cstst::Matrix{T}, Cstel::Vector{T},
    rng::Union{MersenneTwister, Threefry4x}) where T <: Real

    ## here C is the **Covariance** matrix

    μ = Cstel .* f_ell ./ Celel
    Σ = Symmetric(Cstst - Cstel * Cstel' ./ Celel)

    return rand(rng, Distributions.MvNormal( μ, Σ ) )
end
function rfullcond_fstar(f_ell::Vector{T}, Celel::Matrix{T},
    Cstst::T, Cstel::Vector{T},
    rng::Union{MersenneTwister, Threefry4x}, tol=1.0e-10) where T <: Real

    ## here C is the **Covariance** matrix

    μ = Cstel' * ( Celel \ f_ell )
    Σ = Cstst - Cstel' * ( Celel \ Cstel )

    if Σ < 0.0 && abs(Σ) < tol
        Σ = abs(Σ)
    end

    return sqrt(Σ) * randn(rng) + μ
end
function rfullcond_fstar(f_ell::Vector{T}, Celel::Matrix{T},
    Cstst::Matrix{T}, Cstel::Matrix{T},
    rng::Union{MersenneTwister, Threefry4x}) where T <: Real

    ## here C is the **Covariance** matrix

    μ = Cstel * ( Celel \ f_ell )
    Σ = Symmetric(Cstst - Cstel * ( Celel \ Cstel' ))

    return rand(rng, Distributions.MvNormal( μ, Σ ) )
end


### worker function that operates on its own
function update_mixcomp!(mixcomp::MixComponentNormal, prior::PriorMixcomponent_Normal,
    y::Vector{T}, update::Vector{Symbol}=[:μ, :σ2, :κ, :corParams, :fx]) where T <: Real

    nζ = length(mixcomp.ζon_indx)

    if (:κ in update) || (:corParams in update)
        suffstat = update_Cov!(mixcomp, y, prior, SNR_Hyper_ScInvChiSq(), MaternHyper_ScInvChiSq())
    elseif nζ > 0
        W_now, sumWinv_now = getW(mixcomp.κ, mixcomp.D[mixcomp.ζon_indx, mixcomp.ζon_indx], mixcomp.corParams)
        yhat_now, s2_now = getSufficients_W(y[mixcomp.ζon_indx], W_now, sumWinv_now)
        suffstat = Dict(:sumWinv => sumWinv_now, :yhat => yhat_now, :s2 => s2_now)
    end

    if nζ > 0

        if (:μ in update) || (:σ2 in update)
            update_μ_σ2!(mixcomp, suffstat, prior)
        end

        if (:fx in update)
            mixcomp.fx[mixcomp.ζon_indx] = rpost_f(y[mixcomp.ζon_indx], mixcomp.μ, mixcomp.σ2,
                mixcomp.κ, Symmetric(mixcomp.Cor[mixcomp.ζon_indx, mixcomp.ζon_indx]), mixcomp.rng)

            ζoff_indx = setdiff( 1:length(y) , mixcomp.ζon_indx )
            Cov = PDMat_adj(mixcomp.Cor .* (mixcomp.κ * mixcomp.σ2))

            mixcomp.fx[ζoff_indx] = rfullcond_fstar(mixcomp.fx[mixcomp.ζon_indx],
                Cov.mat[mixcomp.ζon_indx, mixcomp.ζon_indx], Cov.mat[ζoff_indx, ζoff_indx],
                Cov.mat[ζoff_indx, mixcomp.ζon_indx], mixcomp.rng)
        end

    else

        if (:μ in update) || (:σ2 in update)
            mixcomp.μ = rand(mixcomp.rng, Normal(prior.μ.μ, sqrt(prior.μ.σ2)))
            mixcomp.σ2 = rand(mixcomp.rng, InverseGamma(0.5*prior.σ2.ν, 0.5*prior.σ2.ν*prior.σ2.s0))
        end

        if (:fx in update)
            Cov = PDMat_adj(mixcomp.Cor .* (mixcomp.κ * mixcomp.σ2))
            mixcomp.fx = rand(mixcomp.rng, Distributions.MvNormal( Cov ) )
        end
    end

    return mixcomp
end


### master function that calls parallel updates
function update_mixcomps!(state::State_GPMTD, prior::Prior_GPMTD, y::Vector{T},
    update::Vector{Symbol}=[:μ, :σ2, :κ, :corParams, :fx]) where T <: Real

    R = length(state.mixcomps)

    for ℓ = 1:R
        state.mixcomps[ℓ].adapt = state.adapt
    end

    state.mixcomps = pmap(update_mixcomp!, state.mixcomps, prior.mixcomps,
        fill(y, R), fill(update, R)) # pmap will update in place on each worker, but we also need to return the mixcomps

    state.accpt = [ deepcopy(state.mixcomps[ℓ].accpt) for ℓ = 1:R ]
    state.adapt_iter = deepcopy(state.mixcomps[1].adapt_iter)

    return nothing
end
