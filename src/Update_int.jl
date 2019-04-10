# Update_int.jl

function update_interc_params!(intercept::InterceptNormal,
    prior::PriorIntercept_Normal, y::Vector{T}, ζ::Vector{Int}) where T <: Real

    ## y is not already subsetted
    indx = findall( ζ .== 0 )
    n = length(indx)
    sumy = sum( y[indx] )

    ## mean
    v1 = 1.0 / ( 1.0 / prior.μ.σ2 + n / intercept.σ2 )
    sd1 = sqrt(v1)
    μ1 = v1 * ( prior.μ.μ / prior.μ.σ2 + sumy / intercept.σ2 )

    intercept.μ = rand( Distributions.Normal(μ1, sd1) )

    ## variance
    a1 = 0.5 * ( prior.σ2.ν + n )
    ss = sum( ( y[indx] .- intercept.μ ).^2 )
    b1 = 0.5 * ( prior.σ2.ν * prior.σ2.s0 + ss )

    intercept.σ2 = rand( Distributions.InverseGamma(a1, b1) )

    return nothing
end
