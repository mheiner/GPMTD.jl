# Update_alloc.jl

export confusion_matrixζ;

function update_ζ!(y::Vector{T}, state::State_GPMTD,
    mixcomp_type::MixComponentNormal) where T <: Real

    n = length(y)
    R = length(state.mixcomps)
    lp = Vector{Float64}(undef, R+1)

    for i = 1:n

        lp[1] = state.lλ[1] - 0.5*log(state.intercept.σ2) -
            0.5 * (y[i] - state.intercept.μ)^2 / state.intercept.σ2

        for j in 1:R
            lp[j+1] = state.lλ[j+1] -
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

function confusion_matrixζ(ζtrue::Vector{Int}, ζest::Vector{Int}, R::Int)

    n = length(ζtrue)
    n2 = length(ζest)

    n == n2 || throw("Two vectors must have same length.")

    out = zeros(Int, R+1, R+1)

    for i = 1:(R+1)
        for j = 1:(R+1)
            out[i,j] = sum( [ζtrue[k] == (i-1) && ζest[k] == (j-1) for k = 1:n] )
        end
    end

    out
end
