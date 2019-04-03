# Update_F.jl

function rpost_f(y::Vector{T}, σ2::T, C::Matrix{T}) where T <: Real
    # conjugate update for MvNormal mean with prior mean 0, known general
    #   covariance C, and indep. noisy obs with variance σ2.

    Cinv = inv(C)
    J = Cinv + I/σ2

    h = (J \ y) ./ σ2

    rand( Distributions.MvNormalCanon(h, J) )
end


function rfullcond_fstar(f_ell::Vector{T}, Celel::Matrix{T},
    Cstst::Matrix{T}, Cstel::Matrix{T}) where T <: Real

    μ = Cstel * ( Celel \ f_ell )
    Σ = Cstst - Cstel * ( Celel \ Cstel' )

    rand( Distributions.MvNormal( μ, Σ ) )
end

function update_F!(F::Matrix{T}, y::Vector{T}, σ2vec::Vector{T},
    Cov::Vector{PDMat}, ζ::Vector{Int}) where T <: Real

    n, R = size(F)
    size(Cov[1]) == (n,n) || throw("update_F! should receive full covariance matrices.")

    nζ = StatsBase.counts(ζ, 1:R) # the 0s don't have a GP
    # Fout = Matrix{T}(undef, n, R) # will update in place

    for ℓ = 1:R  # this could be done in parallel

        if nζ[ℓ] > 0

            ζon = (ζ .== ℓ)
            in_indx = findall( ζon )
            out_indx = findall( .!ζon )

            F[in_indx, ℓ] = rpost_f(y[in_indx], σ2vec[ℓ], Cov[ℓ].mat[in_indx, in_indx])
            F[out_indx, ℓ] = rfullcond_fstar(F[in_indx, ℓ], Cov[ℓ].mat[in_indx, in_indx],
                Cov[ℓ].mat[out_indx, out_indx], Cov[ℓ].mat[out_indx, in_indx])

        else
            F[:, ℓ] = rand( Distributions.MvNormal( Cov[ℓ] ) )
        end

    end

    return nothing
end


function update_σ2vec!(σ2vec::Vector{T}, y::Vector{T}, ζ::Vector{Int},
    F::Matrix{T}, priors::Vector{}) where T <: Real

    n, R = size(F)
    nζ = StatsBase.counts(ζ, 1:R) # the 0s don't have a GP

    for ℓ = 1:R
        ζon = (ζ .== ℓ)
        in_indx = findall( ζon )

        a1 = 0.5 * (priors[ℓ].ν + nζ[ℓ])
        ss = sum( (y[in_indx] - F[in_indx,ℓ]).^2 )
        b1 = 0.5 * (priors[ℓ].ν * priors[ℓ].s0 + ss )

        σ2vec[ℓ] = rand( Distributions.InverseGamma(a1, b1) )
    end

    return nothing
end
