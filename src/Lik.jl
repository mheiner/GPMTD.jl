# Lik.jl

export llik, llik_oos, Qresid!, Qresid_cond!;

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

function llik_oos!(model::Model_GPMTD, postsamp::Dict{Symbol, Any},
                  y_oos::Vector{T}, D_oos::Vector{Array{T, 2}},
                  D_oos_is::Vector{Array{T, 2}}; tol=1.0e-9) where T <: Real

    ## Modifes the state of the rng in the model object

    # D_oos[j] = pairDistMat( X_oos[:,[j]] )
    # D_oos_is[j] = pairDistMat( X_oos[:,[j]],  model.X[:,[j]] )

    llik = 0.0
    lp = Vector{Float64}(undef, model.L+1)

    Fstar = [ postsamp[:mixcomps][j][:μ] .+ rfullcond_fstar!(postsamp[:mixcomps][j][:fx],
        covMat(model.D[j], postsamp[:mixcomps][j][:κ]*postsamp[:mixcomps][j][:σ2], postsamp[:mixcomps][j][:corParams]),
        covMat(D_oos[j], postsamp[:mixcomps][j][:κ]*postsamp[:mixcomps][j][:σ2], postsamp[:mixcomps][j][:corParams]),
        covMat(D_oos_is[j], postsamp[:mixcomps][j][:κ]*postsamp[:mixcomps][j][:σ2], postsamp[:mixcomps][j][:corParams]),
        model.state.mixcomps[j].rng, tol) for j = 1:model.L ]

    for i = 1:length(y_oos)
        lp[1] = postsamp[:lλ][1] - 0.5*log(2π*postsamp[:intercept].σ2) -
                0.5 * (y_oos[i] - postsamp[:intercept].μ)^2 / postsamp[:intercept].σ2

        for j in 1:model.L
            lp[j+1] = postsamp[:lλ][j+1] -
                    0.5*log(2π*postsamp[:mixcomps][j][:σ2]) -
                    0.5 * (y_oos[i] - Fstar[j][i])^2 / postsamp[:mixcomps][j][:σ2]
        end

        llik += SparseProbVec.logsumexp(lp)
    end

    return llik
end


function Qresid!(model::Model_GPMTD, sims::Vector{Any})
    ### Add quantile residuals (conditional model CDF evaluated at each observation) to posterior simulations

    nsim = length(sims)
    lCDF = Vector{Float64}(undef, model.n)
    lP = Vector{Float64}(undef, model.L+1)

    for ii = 1:nsim
        for i = 1:model.n

            lP[1] = sims[ii][:lλ][1] + logcdf( Normal(sims[ii][:intercept].μ, sqrt(sims[ii][:intercept].σ2)), model.y[i] )

            for j in 1:model.L
                lP[j+1] = sims[ii][:lλ][j+1] + logcdf( Normal(sims[ii][:mixcomps][j][:μ] + sims[ii][:mixcomps][j][:fx][i],
                                                               sqrt(sims[ii][:mixcomps][j][:σ2])),
                                                        model.y[i] )
            end

            lCDF[i] = SparseProbVec.logsumexp(lP)
        end

        sims[ii][:CDF] = exp.(lCDF)
    end

    return nothing
end

function Qresid_cond!(model::Model_GPMTD, sims::Vector{Any})
    ### Add quantile residuals (conditional component CDF evaluated at each observation) to posterior simulations

    nsim = length(sims)
    lCDF = Vector{Float64}(undef, model.n)
    lP = 0.0

    for ii = 1:nsim
        for i = 1:model.n

            j = deepcopy(sims[ii][:ζ][i])
            if sims[ii][:ζ][i] == 0
                lP = logcdf( Normal(sims[ii][:intercept].μ, sqrt(sims[ii][:intercept].σ2)), model.y[i] )
            else
                lP = logcdf( Normal(sims[ii][:mixcomps][j][:μ] + sims[ii][:mixcomps][j][:fx][i],
                                                               sqrt(sims[ii][:mixcomps][j][:σ2])),
                                                        model.y[i] )
            end

            lCDF[i] = deepcopy(lP)
        end

        sims[ii][:condCDF] = exp.(lCDF)
    end

    return nothing
end
