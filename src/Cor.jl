# corFun.jl

export MaternParams, corr, corMat, covMat, PDMat_adj;

struct MaternParams
    ν::Real
    lenscale::Real
end

function corr(d::T, params::MaternParams; logout::Bool=false) where T <: Real
    ## Matern

    ν = params.ν
    lenscale = params.lenscale

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

function corrMat(D::Matrix{T}, params::MaternParams; logout::Bool=false) where T <: Real
    if issymmetric(D)

        n = size(D,1)
        C = Matrix{T}(undef, n, n)

        for i = 1:n
            C[i,i] = corr(D[i,i], params, logout=logout)
            for j = (i+1):n
                C[i,j] = corr(D[i,j], params, logout=logout)
                C[j,i] = C[i,j]
            end
        end

    else

        n1, n2 = size(D)
        C = Matrix{T}(undef, n1, n2)

        for i = 1:n1
            for j = 1:n2
                C[i,j] = corr(D[i,j], params, logout=logout)
            end
        end

    end

    return C
end

function covMat(D::Matrix{T}, variance::T, params::MaternParams; tol::Float64=1.0e-6) where T <: Real
    variance > 0.0 || throw("Variance must be positive.")
    lv = log(variance)

    if issymmetric(D)

        n = size(D,1)
        C = Matrix{T}(undef, n, n)

        for i = 1:n
            C[i,i] = exp(lv + corr(D[i,i], params, logout=true))
            for j = (i+1):n
                C[i,j] =  exp(lv + corr(D[i,j], params, logout=true))
                C[j,i] = C[i,j]
            end
        end

        C = PDMat_adj(C, tol)

    else

        n1, n2 = size(D)
        C = Matrix{T}(undef, n1, n2)

        for i = 1:n1
            for j = 1:n2
                C[i,j] =  exp(lv + corr(D[i,j], params, logout=true))
            end
        end

    end

    return C
end

function PDMat_adj(A::Matrix{Float64}, maxadd::Float64=1.0e-6,
    epsfact::Float64=100.0, cumadd::Float64=0.0)

    try PDMat(A)
    catch excep
        if isa(excep, PosDefException) && cumadd <= maxadd
            a = epsfact * eps(Float64)
            A += a * I
            cumadd += a
            epsfactnext = 10.0 * epsfact
            PDMat_adj(A, maxadd, epsfactnext, cumadd)
        else
            PDMat(A) # just trigger original error
        end
    end
end
