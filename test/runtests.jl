using GPMTD
using Test

# @test hello("Julia") == "Hello, Julia"
# @test domath(2.0) ≈ 7.0

## Assure that Distances doesn't change
x = [0.0, 1.0, 1.0]
a = [[0.0, 1.0, 1.0] [1.0, 0.0, 0.0] [1.0, 0.0, 0.0]]
@test pairDistMat(x) ≈ a
@test pairDistMat( hcat(x,x) ) ≈ a .* sqrt(2.0)


## Covariance
x[3] = -1.0

mp = MaternParams(Inf, 1.0)
D = pairDistMat(x)

corr(0.0, mp)
corr(1.0, mp)

corrMat(D, mp)

v = 5.0
Cov = covMat(D, v, mp)

## Log-likelihood calculation
n = 50
σ2 = 2.0
y = 5.0*randn(n)
x = 1.0*randn(n)
D = pairDistMat(x)
v = 5.0
Cov = covMat(D, v, mp, tol=1.0e-10)

x2 = randn(Int(floor(n/2)))
Dxy = pairDistMat(x, x2)
CovXY = covMat(Dxy, v, mp, tol=1.0e-10)

# @benchmark llik_marg($y, $σ2, $Cov) ## winner
# @benchmark logpdf( MvNormal( $Cov + $σ2*I ), $y)

rpost_MvNNV(y, σ2, Cov)
