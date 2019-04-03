# update_Cov.jl


function update_Cov!(y::Vector{T}, σ2vec::Vector{T},
    Cov::Vector{PDMat}, ζ::Vector{Int}, D::Matrix{T}) where T <: Real

    R = length(Cov)

    nζ = StatsBase.counts(ζ, 1:R) # the 0s don't have a GP
    # Fout = Matrix{T}(undef, n, R) # will update in place

    for ℓ = 1:R  # this could be done in parallel

        if nζ[ℓ] > 0

            ζon = (ζ .== ℓ)
            indx = findall( ζon )

            llik_old = llik_marg( y[indx], σ2vec[ℓ], PDMat(Cov[ℓ].mat[indx, indx]) )
            lpri_old

            cand
            Cov_cand  # full for all data points

            llik_new = llik_marg( y[indx], σ2vec[ℓ], PDMat(Cov[ℓ].mat[indx, indx]) )
            lpri_new

            lar = llik_new + lpri_new - llik_old - lpri_old

            if log(rand()) < lar

            end

        else # if no observations assigned to lag ℓ

            new
            Cov[ℓ]

        end

    end

    return nothing
end
