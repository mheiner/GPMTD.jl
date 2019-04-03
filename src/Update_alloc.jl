# Update_alloc.jl

function rpost_ζ_mtdg(S::Vector{Int}, TT::Int,
  lλ::Vector{Float64}, lQ0::Vector{Float64}, lQ::Vector{Matrix{Float64}},
  R::Int, K::Int)

  ζ_out = Vector{Int}(undef, TT-R)

  for i in 1:(TT-R)
      tt = i + R
      Slagrev_now = S[range(tt-1, step=-1, length=R)]
      lp = Vector{Float64}(undef, R+1)

      lp[1] = lλ[1] + lQ0[S[tt]]
      for j in 1:R
          lp[j] = lλ[j+1] + lQ[j][ append!([deepcopy(S[tt])], deepcopy(Slagrev_now[j]))... ]
      end

      w = exp.( lp .- maximum(lp) )
      ζ_out[i] = StatsBase.sample(Weights(w)) - 1 # so that the values are 0:R
  end

  ζ_out
end
