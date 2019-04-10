# Update_alloc.jl

function update_ζ!(y::Vector{T}, state::State_GPMTD, mixcomp_type::MixComponentNormal)

    n = length(y)
    R = length(state.mixcomps)
    lp = Vector{Float64}(undef, R+1)

    for i = 1:n

        lp[1] = lλ[1] - 0.5*log(state.intercept.σ2) -
            0.5 * (y[i] - state.intercept.μ)^2 / state.intercept.σ2

        for j in 1:R
            lp[j] = lλ[j+1] -
                0.5*log(state.mixcomps[j].σ2) -
                0.5 * (y[i] - state.mixcomps[j].μ - state.mixcomps[j].fx[i])^2 / state.mixcomps[j].σ2
        end

        w = exp.( lp .- maximum(lp) )
        state.ζ[i] = StatsBase.sample(Weights(w)) - 1 # so that the values are 0:R

    end

    for ℓ = 1:R
        state.mixcomps[ℓ].ζon_indx = findall( state.ζ .== ℓ )
    end

    return nothing
end
