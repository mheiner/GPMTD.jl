# Sim.jl

export simGPMTD;

function simGPMTD(n::Int, nburn::Int, intercept::InterceptNormal,
    mixcomps::Vector{MixComponentNormal},
    λ::Vector{T}) where T <: Real

    R = length(mixcomps)
    y = zeros(Float64, n + nburn + R)
    xx = [ Vector{Float64}(undef, 0) for ℓ = 1:R ]
    nζ = zeros(Int, R+1)

    for t = (R + 1):(R + nburn + n)

        ζ = StatsBase.sample(0:R, Weights(λ))
        nζ[ζ+1] += 1

        if ζ == 0

            y[t] = sqrt(intercept.σ2)*randn() + intercept.μ

        else

            if nζ[ζ+1] == 0

                f = sqrt(mixcomps[ζ].κ * mixcomps[ζ].σ2) * randn()
                mixcomps[ζ].fx[1] = deepcopy(f)
                y[t] = sqrt(mixcomps[ζ].σ2)*randn() + mixcomps[ζ].μ + f

            else

                mixcomps[ζ].D = expanD(mixcomps[ζ].D, y[t-ζ], x_old)
                mixcomps[ζ].Cor = corrMat(mixcomps[ζ].D, mixcomps[ζ].corParams)

                Cov = PDMat_adj((mixcomps[ζ].κ * mixcomps[ζ].σ2) .* mixcomps[ζ].Cor)
                ncov = size(Cov.mat,1)
                f = rfullcond_fstar(mixcomps[ζ].fx, Cov.mat[2:ncov, 2:ncov],
                    Cov.mat[1,1], Cov.mat[1,2:ncov])
                pushfirst!(mixcomps[ζ].fx, f)

                y[t] = sqrt(mixcomps[ζ].σ2)*randn() + mixcomps[ζ].μ + f

            end

            pushfirst!(xx[ζ], y[t-ζ])

        end
    end

    yout = y[(R + nburn + 1):(R + nburn + n)]
    fx = [ deepcopy(mixcomps.fx) for ℓ = 1:R ]

    return (yout, xx, fx)
end
