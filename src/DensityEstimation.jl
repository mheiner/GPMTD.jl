# DensityEstimation.jl

export getEyPy;


function getEyPy(sims::Array, X_grid::Matrix{T},
    X::Matrix{T}, D::Vector{Matrix{T}}; densout::Bool=false, y_grid::Vector{T}=[0.0],
    rng::MersenneTwister=MersenneTwister()) where T <: Real

    n_star, L = size(X_grid)
    n_y = length(y_grid)

    L == (length(sims[1][:lλ])-1) == length(D) == size(X,2) || throw("L is inconsistent in one of the args.")

    nsim = length(sims)

    Dstar = [ pairDistMat( X_grid[:,[j]] ) for j = 1:L ]
    Dstdat = [ pairDistMat( X_grid[:,[j]],  X[:,[j]] ) for j = 1:L ]

    Fstar = Array{Float64}(undef, L, n_star, nsim)

    for j = 1:L
        Fstar[j,:,:] = hcat([ sims[ii][:mixcomps][j][:μ] .+ rfullcond_fstar(sims[ii][:mixcomps][j][:fx],
        covMat(D[j], sims[ii][:mixcomps][j][:κ]*sims[ii][:mixcomps][j][:σ2], sims[ii][:mixcomps][j][:corParams]),
        covMat(Dstar[j], sims[ii][:mixcomps][j][:κ]*sims[ii][:mixcomps][j][:σ2], sims[ii][:mixcomps][j][:corParams]),
        covMat(Dstdat[j], sims[ii][:mixcomps][j][:κ]*sims[ii][:mixcomps][j][:σ2], sims[ii][:mixcomps][j][:corParams]),
        rng) for ii = 1:nsim ]...)
    end

    Ey = Matrix{Float64}(undef, n_star, nsim)
    if densout
        Py = Array{Float64}(undef, n_y, n_star, nsim)
    end

    for ii = 1:nsim
        for i = 1:n_star

            λ = exp.(sims[ii][:lλ])
            Ey[i,ii] = λ[1] * sims[ii][:intercept].μ + sum( λ[2:(L+1)] .* Fstar[:,i,ii] )

            if densout
                lp = Vector{Float64}(undef, L+1)
                for iii = 1:n_y
                    lp[1] = sims[ii][:lλ][1] - 0.5*log(2π * sims[ii][:intercept].σ2) - 0.5*( y_grid[iii] - sims[ii][:intercept].μ )^2 / sims[ii][:intercept].σ2
                    for j = 1:L
                        lp[j+1] = sims[ii][:lλ][j+1] - 0.5*log(2π * sims[ii][:mixcomps][j][:σ2]) - 0.5*( y_grid[iii] - Fstar[j,i,ii] )^2 / sims[ii][:mixcomps][j][:σ2]
                    end
                    Py[iii,i,ii] = exp( logsumexp(lp) )
                end
            end
        end
    end

    if densout
        return Ey, Py
    else
        return Ey
    end

end
