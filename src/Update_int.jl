# Update_int.jl

function update_interc_params!(μ::T, σ2::T, y::Vector{T}, ζ::Vector{Int},
    prior_μ::, prior_σ2::) where T <: Real

    ## y is not already subsetted
    indx = findall( ζ .== 0 )
    n = length(indx)
    sumy = sum( y[indx] )

    ## mean
    v1 = 1.0 / ( 1.0 / prior_μ.v0 + n / σ2 )
    sd1 = sqrt(v1)
    μ1 = v1 * ( prior_μ.μ0 / v0 + sumy / σ2 )

    μ = rand( Distributions.Normal(μ1, sd1) )

    ## variance
    a1 = 0.5 * ( prior_σ2.ν + n )
    ss = sum( ( y[indx] .- μ ).^2 )
    b1 = 0.5 * ( prior_σ2.ν * prior_σ2.s0 + ss )

    σ2 = rand( Distributions.InverseGamma(a1, b1) )

    return nothing
end
