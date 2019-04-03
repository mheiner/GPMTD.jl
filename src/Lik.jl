# Lik.jl

function llik_marg(y::Vector{T}, σ2::T, Cov::PDMat) where T <: Real
    ## assumes mean zero
    n = length(y)
    Σ = Cov + σ2*I
    ldetΣ = logdet(Σ)
    Σinvy = Σ \ y

    -0.5 * (n*log(2π) + ldetΣ + y'Σinvy )
end

## naive is faster than:
# logpdf( MvNormal( Cov + σ2*I ), y)
