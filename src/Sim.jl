# Sim.jl

export simGPMTD!, simGPMTD_full!;

function simGPMTD!(n::Int, nburn::Int, intercept::InterceptNormal,
    mixcomps::Vector{MixComponentNormal},
    λ::Vector{T}) where T <: Real

    R = length(mixcomps)
    y = zeros(Float64, n + nburn + R)
    ζvec = zeros(Int, n + nburn + R)
    y[1:R] = sqrt(intercept.σ2 / 100.0) .* randn(R)
    xx = [ Matrix{Float64}(undef, 0, 1) for ℓ = 1:R ]
    nζ = zeros(Int, R+1)

    for t = (R + 1):(R + nburn + n)

        ζ = StatsBase.sample(0:R, Weights(λ))

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
                f = rfullcond_fstar(mixcomps[ζ].fx, Cov.mat[2:ncov, 2:ncov],
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

    yout = y[(R + nburn + 1):(R + nburn + n)]
    ζout = ζvec[(R + nburn + 1):(R + nburn + n)]
    fx = [ deepcopy(mixcomps[ℓ].fx) for ℓ = 1:R ]

    return (yout, xx, fx, ζout)
end

function simGPMTD_full!(n::Int, nburn::Int, intercept::InterceptNormal,
    mixcomps::Vector{MixComponentNormal},
    λ::Vector{T}) where T <: Real

    R = length(mixcomps)
    y = zeros(Float64, n + nburn + R)
    X = zeros(Float64, n + nburn, R)
    ζvec = zeros(Int, n + nburn + R)
    y[1:R] = sqrt(intercept.σ2 / 100.0) .* randn(R)

    for ℓ = 1:R
        mixcomps[ℓ].fx[1] = sqrt(mixcomps[ℓ].κ * mixcomps[ℓ].σ2) * randn()
        X[1,ℓ] = deepcopy( y[R+1-ℓ] )
    end

    ζ = StatsBase.sample(0:R, Weights(λ))
    y[R+1] = sqrt(mixcomps[ζ].σ2)*randn() + mixcomps[ζ].μ + mixcomps[ζ].fx[1]
    ζvec[R+1] = deepcopy(ζ)

    for t = (R + 2):(R + nburn + n)
        ii = t - R

        for ℓ = 1:R
            X[ii,ℓ] = deepcopy( y[t-ℓ] )
            mixcomps[ℓ].D = expanD(mixcomps[ℓ].D, X[ii,ℓ], X[(ii-1):-1:1,[ℓ]])
            mixcomps[ℓ].Cor = corrMat(mixcomps[ℓ].D, mixcomps[ℓ].corParams)

            Cov = PDMat_adj((mixcomps[ℓ].κ * mixcomps[ℓ].σ2) .* mixcomps[ℓ].Cor)
            ncov = size(Cov.mat,1)
            f = rfullcond_fstar(mixcomps[ℓ].fx, Cov.mat[2:ncov, 2:ncov],
            Cov.mat[1,1], Cov.mat[1,2:ncov], mixcomps[ℓ].rng)
            pushfirst!(mixcomps[ℓ].fx, f)
        end

        ζ = StatsBase.sample(0:R, Weights(λ))
        ζvec[t] = deepcopy(ζ)

        if ζ == 0
            y[t] = sqrt(intercept.σ2)*randn() + intercept.μ
        else
            y[t] = sqrt(mixcomps[ζ].σ2)*randn() + mixcomps[ζ].μ + mixcomps[ζ].fx[1]
        end

    end

    for ℓ = 1:R
        mixcomps[ℓ].fx = deepcopy(reverse(mixcomps[ℓ].fx)[(nburn+1):(nburn+n)])
    end

    yout = y[(R + nburn + 1):(R + nburn + n)]
    ζout = ζvec[(R + nburn + 1):(R + nburn + n)]
    Xout = X[(nburn+1):(nburn+n),:]

    return (yout, Xout, ζout)
end
