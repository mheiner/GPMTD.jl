# mcmc.jl

export timemod!, etr, mcmc!, adapt!, reset_adapt!;

## from Arthur Lui
function deepcopyFields(state::T, fields::Vector{Symbol}) where T
  substate = Dict{Symbol, Any}()

  for field in fields
    substate[field] = deepcopy(getfield(state, field))
  end

  return substate
end

function postSimsInit(n_keep::Int, init_state::Union{State_GPMTD},
    monitor::Vector{Symbol}=[:intercept, :lλ, :μ, :σ2, :κ,
        :κ_hypers, :corParams, :corHypers, :fx])

    L = length(init_state.mixcomps)

    monitor_outer = intersect(monitor, fieldnames(typeof(init_state)))
    monitor_mixcomps = intersect(monitor, fieldnames(typeof(init_state.mixcomps[1])))

    state = deepcopyFields(init_state, monitor_outer)
    state[:mixcomps] = [ deepcopyFields(init_state.mixcomps[j], monitor_mixcomps) for j = 1:L ]

    state[:llik] = 0.0

    sims = [ deepcopy(state) for i = 1:n_keep ]

    return sims
end

## MCMC timing for benchmarks
function timemod!(n::Int64, model::Union{Model_GPMTD}, niter::Int, outfilename::String; tol_posdef=1.0e-9)
    outfile = open(outfilename, "a+")
    write(outfile, "timing for $(niter) iterations each:\n")
    for i in 1:n
        tinfo = @timed mcmc!(model, niter, tol_posdef=tol_posdef)
        write(outfile, "trial $(i), elapsed: $(tinfo[2]) seconds, allocation: $(tinfo[3]/1.0e6) Megabytes\n")
    end
    close(outfile)
end

## estimate time remaining
function etr(timestart::DateTime, n_keep::Int, thin::Int, outfilename::String)
    timeendburn = now()
    durperiter = (timeendburn - timestart).value / 1.0e5 # in milliseconds
    milsecremaining = durperiter * (n_keep * thin)
    estimatedfinish = now() + Dates.Millisecond(Int64(round(milsecremaining)))
    report_file = open(outfilename, "a+")
    write(report_file, "Completed burn-in at $(durperiter/1.0e3*1000.0) seconds per 1000 iterations \n
      $(durperiter/1.0e3/60.0*1000.0) minutes per 1000 iterations \n
      $(durperiter/1.0e3/60.0/60.0*1000.0) hours per 1000 iterations \n
      estimated completion time $(estimatedfinish)")
    close(report_file)
end

function mcmc!(model::Model_GPMTD, n_keep::Int;
    save::Bool=true,
    thin::Int=1,
    n_procs::Int=1,
    report_filename::String="out_progress.txt",
    report_freq::Int=10000,
    update::Vector{Symbol}=[:intercept, :lλ, :ζ, :μ, :σ2,
        :κ, :κ_hypers, :corParams, :corHypers, :fx],
    monitor::Vector{Symbol}=[:intercept, :lλ, :μ, :σ2,
        :κ, :κ_hypers, :corParams, :corHypers, :fx],
    tol_posdef=1.0e-9)

    ## output files
    report_file = open(report_filename, "a+")
    write(report_file, "Commencing MCMC at $(Dates.now()) for $(n_keep * thin) iterations.\n")

    ## split update parameters
    update_outer = intersect(update, fieldnames(typeof(model.state)))
    update_mixcomps = intersect(update, setdiff( fieldnames(typeof(model.state.mixcomps[1])), [:κ_hypers, :corHypers]))
    up_mixcomps = length(update_mixcomps) > 0
    update_hypers = intersect(update, [:κ_hypers, :corHypers])

    ## collect posterior samples
    if save
        sims = postSimsInit(n_keep, model.state, monitor)
        monitor_outer = intersect(monitor, fieldnames(typeof(model.state)))
        monitor_mixcomps = intersect(monitor, fieldnames(typeof(model.state.mixcomps[1])))
    end

    ## for calculating Metropolis acceptance rates
    start_accpt = deepcopy(model.state.accpt)
    prev_accpt = deepcopy(model.state.accpt)
    start_iter = deepcopy(model.state.iter)

    ## sampling
    for i in 1:n_keep
        for j in 1:thin

            if (:ζ in update_outer)
                update_ζ!(model.y, model.state, model.state.mixcomps[1])
            end

            if (:intercept in update_outer)
                update_interc_params!(model.state.intercept,
                    model.prior.intercept, model.y, model.state.ζ)
            end

            if up_mixcomps
                update_mixcomps!(model.state, model.prior, model.y, update_mixcomps, n_procs=n_procs, tol_posdef=tol_posdef)
            end

            if (:lλ in update_outer)
                update_lλ!(model.state, model.prior.λ)
            end

            if (:κ_hypers in update_hypers) || (:corHypers in update_hypers)
                update_cov_hypers!(model.state, model.prior.covhyper)
            end

            model.state.iter += 1
            if model.state.iter % report_freq == 0
                write(report_file, "Iter $(model.state.iter) at $(Dates.now())\n")
                model.state.llik = llik(model)
                write(report_file, "Log-likelihood $(model.state.llik)\n")
                write(report_file, "Current Metropolis acceptance rates: $(float((model.state.accpt - prev_accpt) / report_freq))\n\n")
                prev_accpt = deepcopy(model.state.accpt)
            end

        end

        if save
            for field in monitor_outer
                sims[i][field] = deepcopy(getfield(model.state, field))
            end
            for j = 1:model.L
                for field in monitor_mixcomps
                    sims[i][:mixcomps][j][field] = deepcopy(getfield(model.state.mixcomps[j], field))
                end
            end
            sims[i][:llik] = llik(model)
        end

    end

    model.state.llik = llik(model)

    close(report_file)

    accptr = float(model.state.accpt - start_accpt) ./ float(model.state.iter - start_iter)

    if save
        return (sims, accptr)
    else
        return (model.state.iter, accptr)
    end

end


function adjust_from_accptr(accptr::T, target::T, adjust_bnds::Array{T,1}) where T <: Real
    if accptr < target
        out = (1.0 - adjust_bnds[1]) * accptr / target + adjust_bnds[1]
    else
        out = (adjust_bnds[2] - 1.0) * (accptr - target) / (1.0 - target) + 1.0
    end

    return out
end


function adapt!(model::Model_GPMTD;
    n_iter_collectSS::Int=2000, n_iter_scale::Int=500,
    accptr_bnds::Vector{T}=[0.23, 0.44],
    adjust_bnds::Vector{T}=[0.01, 10.0],
    maxtries::Int=50,
    report_filename::String="out_progress.txt",
    update::Vector{Symbol}=[:intercept, :lλ, :ζ, :μ, :σ2,
            :κ, :κ_hypers, :corParams, :corHypers, :fx],
    tol_posdef=1.0e-9) where T <: Real

    target = (accptr_bnds[2] - accptr_bnds[1]) / 2.0
    d = Int(2)
    collect_scale = 2.38^2 / float(d)

    ## initial runs
    report_file = open(report_filename, "a+")
    write(report_file, "\n\nBeginning Adaptation Phase 1 of 4 (initial scaling) at $(Dates.now())\n\n")
    close(report_file)

    model.state.adapt = false
    reset_adapt!(model)
    tries = 0

    fails = trues(model.L)

    while any(fails)
        tries += 1
        if tries > maxtries
            println("Exceeded maximum adaptation attempts, Phase 1\n")
            report_file = open(report_filename, "a+")
            write(report_file, "\n\nExceeded maximum adaptation attempts, Phase 1\n\n")
            close(report_file)
            break
        end

        iter, accptr = mcmc!(model, n_iter_scale, save=false,
            report_filename=report_filename,
            report_freq=n_iter_scale, update=update, tol_posdef=tol_posdef)

        for j = 1:model.L
            fails[j] = (accptr[j] < accptr_bnds[1])
            if fails[j]
                model.state.mixcomps[j].cSig = PDMat_adj(adjust_from_accptr(accptr[j], target, adjust_bnds) * model.state.mixcomps[j].cSig.mat)
            end
        end

    end


    ## local scaling
    report_file = open(report_filename, "a+")
    write(report_file, "\n\nBeginning Adaptation Phase 2 of 4 (local scaling) at $(Dates.now())\n\n")
    close(report_file)

    model.state.adapt = false
    reset_adapt!(model)

    localtarget = target * 1.0

    for ii = 1:3

        for k = 1:d

            tries = 0
            fails = trues(model.L)
            while any(fails)
                tries += 1
                if tries > maxtries
                    println("Exceeded maximum adaptation attempts, Phase 2\n")
                    report_file = open(report_filename, "a+")
                    write(report_file, "\n\nExceeded maximum adaptation attempts, Phase 2\n\n")
                    close(report_file)
                    break
                end

                iter, accptr = mcmc!(model, n_iter_scale, save=false,
                    report_filename=report_filename,
                    report_freq=n_iter_scale, update=update, tol_posdef=tol_posdef)

                for j = 1:model.L
                    too_low = accptr[j] < (accptr_bnds[1] * 0.5)
                    too_high = accptr[j] > (accptr_bnds[2])

                    if too_low || too_high

                        fails[j] = true

                        tmp = Matrix(model.state.mixcomps[j].cSig)
                        σ = sqrt.(LinearAlgebra.diag(tmp))
                        ρ = StatsBase.cov2cor(tmp, σ)

                        σ[k] *= adjust_from_accptr(accptr[j], localtarget, adjust_bnds)
                        tmp = StatsBase.cor2cov(ρ, σ)
                        tmp += Diagonal(fill(0.1*minimum(σ), size(tmp,1)))
                        model.state.mixcomps[j].cSig = PDMat_adj(tmp)

                    else
                        fails[j] = false
                    end

                end

            end

        end
    end


    ## cΣ collection
    report_file = open(report_filename, "a+")
    write(report_file, "\n\nBeginning Adaptation Phase 3 of 4 (covariance collection) at $(Dates.now())\n\n")
    close(report_file)

    reset_adapt!(model)
    model.state.adapt = true

    iter, accptr = mcmc!(model, n_iter_collectSS, save=false,
        report_filename=report_filename,
        report_freq=1000, update=update, tol_posdef=tol_posdef)

    for j = 1:model.L
        Sighat = model.state.mixcomps[j].runningSS_Met / float(model.state.adapt_iter)
        # minSighat = minimum(abs.(Sighat))
        # SighatPD = Sighat + Matrix(Diagonal(fill(0.1*minSighat, d)))
        model.state.mixcomps[j].cSig = PDMat_adj(collect_scale * Sighat)
    end

    ## final scaling
    report_file = open(report_filename, "a+")
    write(report_file, "\n\nBeginning Adaptation Phase 4 of 4 (final scaling) at $(Dates.now())\n\n")
    close(report_file)

    model.state.adapt = false
    reset_adapt!(model)
    tries = 0

    fails = trues(model.L)

    while any(fails)
        tries += 1
        if tries > maxtries
            println("Exceeded maximum adaptation attempts, Phase 4\n")
            report_file = open(report_filename, "a+")
            write(report_file, "\n\nExceeded maximum adaptation attempts, Phase 4\n\n")
            close(report_file)
            break
        end

        iter, accptr = mcmc!(model, n_iter_scale, save=false,
            report_filename=report_filename,
            report_freq=n_iter_scale, update=update, tol_posdef=tol_posdef)

        for j = 1:model.L
            too_low = accptr[j] < accptr_bnds[1]
            too_high = accptr[j] > accptr_bnds[2]

            if too_low || too_high
                fails[j] = true
                model.state.mixcomps[j].cSig = PDMat_adj(adjust_from_accptr(accptr[j], target, adjust_bnds) * model.state.mixcomps[j].cSig.mat)
            else
                fails[j] = false
            end

        end

    end

    report_file = open(report_filename, "a+")
    write(report_file, "\n\nAdaptation concluded at $(Dates.now())\n\n")
    close(report_file)

    ## note that mcmc! also closes the report file
    reset_adapt!(model)
    model.state.adapt = false

    return nothing
end


function reset_adapt!(model::Model_GPMTD, nparams::Int=2)
    model.state.adapt_iter = 0
    L = length(model.state.mixcomps)
    for j = 1:L
        model.state.mixcomps[j].adapt_iter = 0
        model.state.mixcomps[j].runningsum_Met = zeros( Float64, nparams )
        model.state.mixcomps[j].runningSS_Met = zeros( Float64, nparams, nparams )
    end

    return nothing
end
