# update_Cov.jl

function getW(κ::T, D::Matrix{T}, corParams::CorParams) where T <: Real
    κCor = covMat(D, κ, corParams) # not the inteded use of the function, but this works.
    W = PDMat(κCor + I)
    Winv1 = W \ ones(T, size(W,1))
    sumWinv = sum(Winv1)
    return (W, sumWinv)
end

function getSufficients_W(y::Vector{T}, W::PDMat, sumWinv::T,
    mixcomp_type::MixComponentNormal=MixComponentNormal()) where T <: Real

    yhat = sum(W \ y) / sumWinv
    d = y .- yhat
    Winvd = W \ d
    s2 = d'Winvd

    return (yhat, s2)
end

function lmarg_W(W::PDMat, sumWinv::T, yhat::T, s2::T, σ2::T,
    priorMixcomponent::PriorMixcomponent_Normal) where T <: Real

    a1 = logdet(W)
    a2 = s2 / σ2
    a3 = log(sumWinv)
    a40 = σ2 / sumWinv + priorMixcomponent.μ.σ2
    a4 = log(a40)
    a5 = (yhat - priorMixcomponent.μ.μ)^2 / a40

    return -0.5*( a1 + a2 + a3 + a4 + a5 )
end

function ldens_proptoIG_Jacobian(lx::T, shape::T, scale::T) where T <: Real
    ## Jacobian included for log scale
    x = exp(lx)
    return -shape * lx - scale / x
end

function lprior_κCor(lκ::T, lcorParam::T, κHypers::SNR_Hyper_ScInvChiSq,
    corParamHypers::MaternHyper_ScInvChiSq) where T <: Real

    la = ldens_proptoIG_Jacobian(lκ, 0.5*κHypers.κ_ν, 0.5*κHypers.κ_ν*κHypers.κ0)
    lb = ldens_proptoIG_Jacobian(lcorParam, 0.5*corParamHypers.lenscale_ν,
        0.5*corParamHypers.lenscale_ν*corParamHypers.lenscale0)
    return la + lb
end

function update_Cov!(mixcomp::MixComponentNormal, y::Vector{T}, # assumes y is full
    priorMixcomponent::PriorMixcomponent_Normal,
    κHypers_type::SNR_Hyper_ScInvChiSq, corParamHypers_type::MaternHyper_ScInvChiSq) where T <: Real

    nζon = length(mixcomp.ζon_indx)

    lκ_old = log(mixcomp.κ)
    lcorParam_old = log(mixcomp.corParams.lenscale)

    if nζon > 0

        W_old, sumWinv_old = getW(mixcomp.κ, mixcomp.D[mixcomp.ζon_indx, mixcomp.ζon_indx], mixcomp.corParams)
        yhat_old, s2_old =  getSufficients_W(y[mixcomp.ζon_indx], W_old, sumWinv_old)

        llik_old = lmarg_W(W_old, sumWinv_old, yhat_old, s2_old, mixcomp.σ2, priorMixcomponent)
        lpri_old = lprior_κCor(lκ_old, lcorParam_old, mixcomp.κ_hypers, mixcomp.corHypers)

        cand = [ lκ_old, lcorParam_old ] + rand(mixcomp.rng, MvNormal(mixcomp.cSig.mat))
        κ_cand = exp(cand[1])
        corParams_cand = deepcopy(mixcomp.corParams)
        corParams_cand.lenscale = exp(cand[2])

        W_cand, sumWinv_cand = getW(κ_cand, mixcomp.D[mixcomp.ζon_indx, mixcomp.ζon_indx], corParams_cand)
        yhat_cand, s2_cand =  getSufficients_W(y[mixcomp.ζon_indx], W_cand, sumWinv_cand)

        llik_cand = lmarg_W(W_cand, sumWinv_cand, yhat_cand, s2_cand, mixcomp.σ2, priorMixcomponent)
        lpri_cand = lprior_κCor(cand[1], cand[2], mixcomp.κ_hypers, mixcomp.corHypers)

        lar = llik_cand + lpri_cand - llik_old - lpri_old

        if log(rand(mixcomp.rng)) < lar
            mixcomp.κ = deepcopy(κ_cand)
            mixcomp.corParams = deepcopy(corParams_cand)
            mixcomp.Cor = corrMat(mixcomp.D, mixcomp.corParams)
            mixcomp.accpt += 1

            lparams_out = deepcopy(cand)
            out = Dict(:sumWinv => sumWinv_cand, :yhat => yhat_cand, :s2 => s2_cand)
        else
            lparams_out = vcat(deepcopy(lκ_old), deepcopy(lcorParam_old))
            out = Dict(:sumWinv => sumWinv_old, :yhat => yhat_old, :s2 => s2_old)
        end

    else # if no observations assigned to the updated lag

        lpri_old = lprior_κCor(lκ_old, lcorParam_old, mixcomp.κ_hypers, mixcomp.corHypers)

        cand = [ lκ_old, lcorParam_old ] + rand(mixcomp.rng, MvNormal(mixcomp.cSig.mat))
        κ_cand = exp(cand[1])
        corParams_cand = deepcopy(mixcomp.corParams)
        corParams_cand.lenscale = exp(cand[2])

        lpri_cand = lprior_κCor(cand[1], cand[2], mixcomp.κ_hypers, mixcomp.corHypers)

        lar = lpri_cand - lpri_old

        if log(rand(mixcomp.rng)) < lar
            mixcomp.κ = deepcopy(κ_cand)
            mixcomp.corParams = deepcopy(corParams_cand)
            mixcomp.Cor = corrMat(mixcomp.D, mixcomp.corParams)
            mixcomp.accpt += 1

            lparams_out = deepcopy(cand)
            out = Dict()
        else
            lparams_out = vcat(deepcopy(lκ_old), deepcopy(lcorParam_old))
            out = Dict()
        end
    end

    if mixcomp.adapt
        mixcomp.adapt_iter += 1
        mixcomp.runningsum_Met += lparams_out
        runningmean = mixcomp.runningsum_Met / float(mixcomp.adapt_iter)
        runningdev = ( lparams_out - runningmean )
        mixcomp.runningSS_Met = runningdev * runningdev' # the mean is changing, but this approx. is fine.
    end

    return out
end
