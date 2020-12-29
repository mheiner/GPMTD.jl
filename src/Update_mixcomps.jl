# Update_mixcomps.jl

export rfullcond_fstar!;

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

# function rpost_f!(y::Vector{T}, μ::T, σ2::T, κ::T, Cor::Symmetric{Float64, Matrix{Float64}},
#     rng::Union{MersenneTwister, Threefry4x}) where T <: Real
#     # conjugate update for MvNormal mean with prior mean μ, known general
#     #   correlation matrix Cor, signal-to-noise ratio κ, and indep. noisy obs with variance σ2.
#
#     Corinv = inv(Cor)
#     J = PDMat_adj((Corinv ./ κ + I) ./ σ2, 1.0e-3)
#
#     h = ( y .- μ ) ./ σ2
#
#     return rand(rng, Distributions.MvNormalCanon(h, J) )
# end
# function rpost_f!(y::Vector{T}, μ::T, σ2::T, κ::T, Cor::Symmetric{Float64, Matrix{Float64}},
#     rng::Union{MersenneTwister, Threefry4x}) where T <: Real
#     # conjugate update for MvNormal mean with prior mean μ, known general
#     #   correlation matrix Cor, signal-to-noise ratio κ, and indep. noisy obs with variance σ2.
#
#     J = inv(PDMat_adj(κ*σ2*Cor)) + I/σ2
#     h = ( y .- μ ) ./ σ2
#
#     return rand(rng, Distributions.MvNormalCanon(h, J) )
# end
function rpost_f!(y::Vector{T}, μ::T, σ2::T, κ::T, Cor::Symmetric{Float64, Matrix{Float64}},
    rng::Union{MersenneTwister, Threefry4x}) where T <: Real
    # conjugate update for MvNormal mean with prior mean μ, known general
    #   correlation matrix Cor, signal-to-noise ratio κ, and indep. noisy obs with variance σ2.

    # Rasmussen & Williams '06, p.46
    K = κ * Cor
    K1 = (K + I) \ K
    Σ0 = K * (I - K1)
    Σ1 = PDMat_adj(Symmetric( σ2*Σ0 ))
    μ1 = Σ1 * ( y .- μ ) ./ σ2

    return rand(rng, Distributions.MvNormal(μ1, Σ1) )
end


function rfullcond_fstar!(f_ell::T, Celel::T, Cstst::T, Cstel::T,
    rng::Union{MersenneTwister, Threefry4x}, tol=1.0e-10) where T <: Real

    ## here C is the **Covariance** matrix

    μ = Cstel * ( Celel \ f_ell )
    Σ = Cstst - Cstel * ( Celel \ Cstel )

    if Σ < 0.0 && abs(Σ) < tol
        Σ = abs(Σ)
    end

    return sqrt(Σ) * randn(rng) + μ
end
function rfullcond_fstar!(f_ell::T, Celel::T,
    Cstst::Matrix{T}, Cstel::Vector{T},
    rng::Union{MersenneTwister, Threefry4x}, tol=1.0e-10) where T <: Real

    ## here C is the **Covariance** matrix

    μ = Cstel .* f_ell ./ Celel
    Σ = PDMat_adj(Symmetric(Cstst - Cstel * Cstel' ./ Celel), tol)

    return rand(rng, Distributions.MvNormal( μ, Σ ) )
end
function rfullcond_fstar!(f_ell::Vector{T}, Celel::Matrix{T},
    Cstst::T, Cstel::Vector{T},
    rng::Union{MersenneTwister, Threefry4x}, tol=1.0e-10) where T <: Real

    ## here C is the **Covariance** matrix

    Celel = PDMat_adj(Celel, tol)

    μ = Cstel' * ( Celel \ f_ell )
    Σ = Cstst - Cstel' * ( Celel \ Cstel )

    if Σ < 0.0 && abs(Σ) < tol
        Σ = abs(Σ)
    end

    return sqrt(Σ) * randn(rng) + μ
end
# function rfullcond_fstar!(f_ell::Vector{T}, Celel::Matrix{T},
#     Cstst::Matrix{T}, Cstel::Matrix{T},
#     rng::Union{MersenneTwister, Threefry4x}) where T <: Real
#
#     ## here C is the **Covariance** matrix
#
#     μ = Cstel * ( Celel \ f_ell )
#     Σ = PDMat_adj(Symmetric(Cstst - Cstel * ( Celel \ Cstel' )))
#
#     return rand(rng, Distributions.MvNormal( μ, Σ ) )
# end
function rfullcond_fstar!(f_ell::Vector{T}, Celel::Matrix{T},
    Cstst::Matrix{T}, Cstel::Matrix{T},
    rng::Union{MersenneTwister, Threefry4x}, tol=1.0e-10) where T <: Real

    ## here C is the **Covariance** matrix

    Celel = PDMat_adj(Celel, tol)

    μ = Cstel * ( Celel \ f_ell )
    Σ = PDMat_adj(Symmetric(Cstst - X_invA_Xt(Celel, Cstel) ), tol)

    return rand(rng, Distributions.MvNormal( μ, Σ ) )
end


# function rfullcond_fstar!_canon(fx::Vector{T}, ## already indexed
#     Cov::PDMat, on_indx::Vector{Int}, off_indx::Vector{Int},
#     rng::Union{MersenneTwister, Threefry4x}) where T <: Real
#
#     ## assumes zero mean GP
#
#     Λ = inv(Cov)
#     J = PDMat_adj( Λ.mat[off_indx, off_indx] )
#
#     η = - vec( Λ.mat[off_indx, on_indx] * fx )
#
#     return rand(rng, Distributions.MvNormalCanon( η, J ) )
# end


### worker function that operates on its own
function update_mixcomp!(mixcomp::MixComponentNormal, prior::PriorMixcomponent_Normal,
    y::Vector{T}, update::Vector{Symbol}=[:μ, :σ2, :κ, :corParams, :fx], tol_posdef=1.0e-9) where T <: Real

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
            mixcomp.fx[mixcomp.ζon_indx] = rpost_f!(y[mixcomp.ζon_indx], mixcomp.μ, mixcomp.σ2,
                mixcomp.κ, Symmetric(mixcomp.Cor[mixcomp.ζon_indx, mixcomp.ζon_indx]), mixcomp.rng)

            ζoff_indx = setdiff( 1:length(y) , mixcomp.ζon_indx )
            Cov = PDMat_adj(mixcomp.Cor .* (mixcomp.κ * mixcomp.σ2))

            # mixcomp.fx[ζoff_indx] = rfullcond_fstar_canon(mixcomp.fx[mixcomp.ζon_indx], Cov,
            #     mixcomp.ζon_indx, ζoff_indx, mixcomp.rng)

            mixcomp.fx[ζoff_indx] = rfullcond_fstar!(mixcomp.fx[mixcomp.ζon_indx],
                Cov.mat[mixcomp.ζon_indx, mixcomp.ζon_indx], Cov.mat[ζoff_indx, ζoff_indx],
                Cov.mat[ζoff_indx, mixcomp.ζon_indx], mixcomp.rng, tol_posdef)
        end

    else

        if (:μ in update) || (:σ2 in update)
            mixcomp.μ = rand(mixcomp.rng, Normal(prior.μ.μ, sqrt(prior.μ.σ2)))
            mixcomp.σ2 = rand(mixcomp.rng, InverseGamma(0.5*prior.σ2.ν, 0.5*prior.σ2.ν*prior.σ2.s0))
        end

        if (:fx in update)
            Cov = PDMat_adj(mixcomp.Cor .* (mixcomp.κ * mixcomp.σ2), tol_posdef)
            mixcomp.fx = rand(mixcomp.rng, Distributions.MvNormal( Cov.mat ) )
        end
    end

    return mixcomp
end


### master function that calls parallel updates
function update_mixcomps!(state::State_GPMTD, prior::Prior_GPMTD, y::Vector{T},
    update::Vector{Symbol}=[:μ, :σ2, :κ, :corParams, :fx];
    n_procs::Int=1, tol_posdef=1.0e-9) where T <: Real

    L = length(state.mixcomps)

    for ℓ = 1:L
        state.mixcomps[ℓ].adapt = state.adapt
    end

    if n_procs == 1
        for ℓ = 1:L
            state.mixcomps[ℓ] = update_mixcomp!(state.mixcomps[ℓ], prior.mixcomps[ℓ], y, update, tol_posdef)
        end
    elseif n_procs > 1
        state.mixcomps = pmap(update_mixcomp!, state.mixcomps, prior.mixcomps,
            fill(y, L), fill(update, L), fill(tol_posdef, L)) # pmap will update in place on each worker, but we also need to return the mixcomps
    end

    state.accpt = [ deepcopy(state.mixcomps[ℓ].accpt) for ℓ = 1:L ]
    state.adapt_iter = deepcopy(state.mixcomps[1].adapt_iter)

    return nothing
end
