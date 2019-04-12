# Sim.jl

export simGPMTD;

function simGPMTD(n::Int, nburn::Int, intercept::InterceptNormal,
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
