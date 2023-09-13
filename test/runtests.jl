using Test
using ITensors
using ITensorAGP
using LinearAlgebra

"""
    matricize(M::MPO)

Construct the matrix array of an MPO.
"""
function matricize(M::MPO)
  sites = siteinds(M; plev=0)
  C = combiner(sites)
  return matrix(prod(M) * C' * C)
end

"""
    S_matrix(A::Matrix, H::Matrix, ∂H::Matrix)

Construct the action of A (that is, the trace of (∂H+i[A,H])^2) using matrix operations.
"""
function S_matrix(A::Matrix, H::Matrix, ∂H::Matrix)
  G = ∂H + 1.0im * (A * H - H * A)
  return real(tr(adjoint(G) * G))
end

"""
    computeAGP(E::Vector, V::Matrix, ∂H::Matrix)

Construct the exact adiabatic gauge potential (AGP).
"""
function computeAGP(E::Vector, V::Matrix, ∂H::Matrix)
  ∂H = adjoint(V) * ∂H * V
  N = length(E)

  Erow = E[reshape(repeat(1:N, N), N, N)]
  AGP = 1.0im ./ (transpose(Erow) - Erow)
  AGP = AGP .* ∂H
  AGP[diagind(AGP)] .= 0.0

  return AGP
end

@testset "ITensorAGP.jl" begin
  include(joinpath(pkgdir(ITensorAGP), "examples", "transverse_field_ising.jl"))
  res = main(; L=8, hz=0.5, outputlevel=0)
  AGP = res.AGP
  ls_error = res.ls_error

  @test ls_error ≈ 0 atol = 1e-3

  H = res.H
  dH = res.dH

  ITensors.disable_warn_order()
  H_matrix = matricize(H)
  dH_matrix = matricize(dH)
  AGP_matrix = matricize(AGP)

  E, V = eigen(H_matrix)
  exactAGP = computeAGP(E, V, dH_matrix)
  AGP_matrix = adjoint(V) * AGP_matrix * V  # same basis as `exactAGP`
  AGP_matrix[diagind(AGP_matrix)] .= 0.0  # gauge freedom

  @test S_matrix(AGP_matrix, H_matrix, dH_matrix) ≈ S_matrix(exactAGP, H_matrix, dH_matrix) rtol =
    1e-3
end
