# corFun.jl

function corMatern(d::Float64, ν::Float64, lenscale::Float64; logout::Bool=false)

    d >= 0.0 || throw(error("Distance d must be non-negative."))
    lenscale > 0.0 || throw(error("Lengthscale parameter lenscale must be positive."))

    if ν == 0.5

        lcor = -d / lenscale

    elseif ν == 1.5

        s3ddl = sqrt(3.0) * d / lenscale
        lcor = log( s3ddl  + 1.0 ) - s3ddl

    elseif ν == 2.5

        s5ddl = sqrt(5.0) * d / lenscale
        lcor = log( s5ddl^2 / 3.0 + s5ddl + 1.0 ) - s5ddl

    elseif ν == Inf  # squared exponential correlation function

        lcor = -0.5 * ( d / lenscale )^2

    else
        throw(error("Unsupported value of ν."))
    end

    logout ? out = lcor : out = exp(lcor)
    return out
end
