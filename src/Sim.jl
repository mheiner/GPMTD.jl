# Sim.jl

export simGPMTD!, simGPMTD_full!, forecast_sim!;

function simGPMTD!(n::Int, nburn::Int, intercept::InterceptNormal,
    mixcomps::Vector{MixComponentNormal},
    λ::Vector{T}) where T <: Real

    L = length(mixcomps)
    y = zeros(Float64, n + nburn + L)
    ζvec = zeros(Int, n + nburn + L)
    y[1:L] = sqrt(intercept.σ2 / 100.0) .* randn(L)
    xx = [ Matrix{Float64}(undef, 0, 1) for ℓ = 1:L ]
    nζ = zeros(Int, L+1)

    for t = (L + 1):(L + nburn + n)

        ζ = StatsBase.sample(0:L, Weights(λ))

        if ζ == 0

            y[t] = sqrt(intercept.σ2)*randn() + intercept.μ

        else

            if nζ[ζ+1] == 0

                f = sqrt(mixcomps[ζ].κ * mixcomps[ζ].σ2) * randn()
                mixcomps[ζ].fx[1] = deepcopy(f)
                y[t] = sqrt(mixcomps[ζ].σ2)*randn() + mixcomps[ζ].μ + f

            else

                mixcomps[ζ].D = expanD(mixcomps[ζ].D, y[t-ζ], xx[ζ])
                mixcomps[ζ].Cor = corrMat(mixcomps[ζ].D, mixcomps[ζ].corParams)

                Cov = PDMat_adj((mixcomps[ζ].κ * mixcomps[ζ].σ2) .* mixcomps[ζ].Cor)
                ncov = size(Cov.mat,1)
                f = rfullcond_fstar!(mixcomps[ζ].fx, Cov.mat[2:ncov, 2:ncov],
                    Cov.mat[1,1], Cov.mat[1,2:ncov], mixcomps[ζ].rng)
                pushfirst!(mixcomps[ζ].fx, f)

                y[t] = sqrt(mixcomps[ζ].σ2)*randn() + mixcomps[ζ].μ + f

            end

            # pushfirst!(xx[ζ][:,1], y[t-ζ])
            xx[ζ] = reshape(vcat(y[t-ζ], xx[ζ][:,1]), size(xx[ζ],1)+1, 1)
            nζ[ζ+1] += 1
            ζvec[t] = deepcopy(ζ)

        end
    end

    yout = y[(L + nburn + 1):(L + nburn + n)]
    ζout = ζvec[(L + nburn + 1):(L + nburn + n)]
    fx = [ deepcopy(mixcomps[ℓ].fx) for ℓ = 1:L ]

    return (yout, xx, fx, ζout)
end

function simGPMTD_full!(n::Int, nburn::Int, intercept::InterceptNormal,
    mixcomps::Vector{MixComponentNormal},
    λ::Vector{T}) where T <: Real

    L = length(mixcomps)
    y = zeros(Float64, n + nburn + L)
    X = zeros(Float64, n + nburn, L)
    ζvec = zeros(Int, n + nburn + L)
    y[1:L] = sqrt(intercept.σ2 / 100.0) .* randn(L)

    for ℓ = 1:L
        mixcomps[ℓ].fx[1] = sqrt(mixcomps[ℓ].κ * mixcomps[ℓ].σ2) * randn()
        X[1,ℓ] = deepcopy( y[L+1-ℓ] )
    end

    ζ = StatsBase.sample(0:L, Weights(λ))
    y[L+1] = sqrt(mixcomps[ζ].σ2)*randn() + mixcomps[ζ].μ + mixcomps[ζ].fx[1]
    ζvec[L+1] = deepcopy(ζ)

    for t = (L + 2):(L + nburn + n)
        ii = t - L

        for ℓ = 1:L
            X[ii,ℓ] = deepcopy( y[t-ℓ] )
            mixcomps[ℓ].D = expanD(mixcomps[ℓ].D, X[ii,ℓ], X[(ii-1):-1:1,[ℓ]])
            mixcomps[ℓ].Cor = corrMat(mixcomps[ℓ].D, mixcomps[ℓ].corParams)

            Cov = PDMat_adj((mixcomps[ℓ].κ * mixcomps[ℓ].σ2) .* mixcomps[ℓ].Cor)
            ncov = size(Cov.mat,1)
            f = rfullcond_fstar!(mixcomps[ℓ].fx, Cov.mat[2:ncov, 2:ncov],
                                 Cov.mat[1,1], Cov.mat[1,2:ncov], mixcomps[ℓ].rng)
            pushfirst!(mixcomps[ℓ].fx, f)
        end

        ζ = StatsBase.sample(0:L, Weights(λ))
        ζvec[t] = deepcopy(ζ)

        if ζ == 0
            y[t] = sqrt(intercept.σ2)*randn() + intercept.μ
        else
            y[t] = sqrt(mixcomps[ζ].σ2)*randn() + mixcomps[ζ].μ + mixcomps[ζ].fx[1]
        end

    end

    for ℓ = 1:L
        mixcomps[ℓ].fx = deepcopy(reverse(mixcomps[ℓ].fx)[(nburn+1):(nburn+n)])
    end

    yout = y[(L + nburn + 1):(L + nburn + n)]
    ζout = ζvec[(L + nburn + 1):(L + nburn + n)]
    Xout = X[(nburn+1):(nburn+n),:]

    return (yout, Xout, ζout)
end

## forecasting

function draw_f_next!(k::Int, DD::Matrix{T}, # XXk::T, XX1tok::Matrix{T},
    cc::T, mxcomp::Dict{Symbol, Any}, mod_mxcomp::MixComponentGPMTD, tol::T) where T <: Real
    ## for easy parallelization; each of these arguments is an element of a vector of length L

    # DD = expanD(DD, XXk, XX1tok, front=false)
    Cov = PDMat_adj(covMat(DD, cc, mxcomp[:corParams]), tol) # redundant, but not expensive
    ncov = size(Cov.mat, 1)

    f = rfullcond_fstar!(mxcomp[:fx], Cov.mat[1:(ncov-1), 1:(ncov-1)],
                         Cov.mat[ncov,ncov], Cov.mat[ncov,1:(ncov-1)], mod_mxcomp.rng, tol) # redundant and expensive, but necessary until we have a numerically stable shortcut

    push!(mxcomp[:fx], f)

    return mxcomp
end


function forecast_sim!(model::Model_GPMTD, K::Int, postsamp::Dict{Symbol, Any}; tol=1.0e-3, n_procs=1)
    ## modifies each rng of model.state.mixcomps

    mxcomp = deepcopy(postsamp[:mixcomps])
    yy = vcat( deepcopy(model.y), zeros(K) )
    XX = vcat( deepcopy(model.X), zeros(K, model.L) )

    λw = Weights(exp.(postsamp[:lλ]))
    ζ_new = [ StatsBase.sample(0:model.L, λw) for k = 1:K ]

    cc = [ mxcomp[j][:κ] * mxcomp[j][:σ2] for j = 1:model.L ]
    DD = [ deepcopy(model.D[j]) for j = 1:model.L ]

    ## appears to be totally unstable
      # C_inv = [ inv(PDMat_adj(covMat(model.D[j], cc[j], mxcomp[j][:corParams]), 1.0e-09).chol) for j = 1:model.L ] # inv(PDMat(C).chol) goes straight to inverse of C

    for k = 1:K
        XX[model.n + k, :] = deepcopy(yy[(model.n - 1 + k):-1:(model.n - model.L + k)])

        ## part of the unstable shortcut
        # dd = [ pairDistMat( XX[[model.n + k], [j]], XX[1:(model.n + k - 1), [j]] ) for j = 1:model.L ]
        # cc_vec = [ vec(covMat(dd[j], cc[j], mxcomp[j][:corParams])) for j = 1:model.L ]
        # sd_f = [ cc[j] - cc_vec[j]'C_inv[j]*cc_vec[j] for j = 1:model.L ]

        # XXk = [ deepcopy(XX[model.n+k, j]) for j = 1:model.L ]
        # XX1tok = [ deepcopy(XX[1:(model.n + k - 1), [j]]) for j = 1:model.L ]
        # DD = [ expanD(DD[j], XXk[j], XX1tok[j], front=false) for j = 1:model.L ]

        DD = [ expanD(DD[j], XX[model.n+k, j], XX[1:(model.n + k - 1), [j]], front=false) for j = 1:model.L ]

        if n_procs == 1
            for j = 1:model.L
                # DD[j] = expanD(DD[j], XXk[j], XX1tok[j], front=false)
                # Cov = PDMat_adj(covMat(DD[j], cc[j], mxcomp[j][:corParams]), tol)
                # ncov = size(Cov.mat, 1)
                #
                # f = rfullcond_fstar!(mxcomp[j][:fx], Cov.mat[1:(ncov-1), 1:(ncov-1)],
                #                      Cov.mat[ncov,ncov], Cov.mat[ncov,1:(ncov-1)], model.state.mixcomps[j].rng, tol)
                # push!(mxcomp[j][:fx], f)

                ## instead of above
                mxcomp[j] = draw_f_next!(k, DD[j], cc[j], mxcomp[j], model.state.mixcomps[j], tol) # apparently can't modify DD if I pass DD[j]?
            end
        elseif n_procs > 1 # parallelization does not solve the issue of large n which can really slow down forecasting, but hopefully speed things up modestly
            mxcomp = pmap(draw_f_next!, fill(k, model.L), DD, cc, mxcomp, model.state.mixcomps, fill(tol, model.L)) # pmap will update in place on each worker
        end

        if ζ_new[k] == 0
            yy[model.n + k] = sqrt(postsamp[:intercept].σ2) * randn() + postsamp[:intercept].μ
        else
            yy[model.n + k] = sqrt(mxcomp[ζ_new[k]][:σ2]) * randn() + mxcomp[ζ_new[k]][:μ] + mxcomp[ζ_new[k]][:fx][model.n + k]
        end
    end

    return yy[(model.n + 1):(model.n + K)]
end

function forecast_sim!(model::Model_GPMTD, K::Int, sims::Vector{Any}; tol=1.0e-3, n_procs=1)
    nsim = length(sims)
    Yforec = permutedims( hcat( [ forecast_sim!(model, K, sims[i], tol=tol, n_procs=n_procs) for i = 1:nsim ]... ) )
    return Yforec
end
