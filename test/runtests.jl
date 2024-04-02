@eval module $(gensym())
using ITensorAGP: ITensorAGP
using ITensors: ITensors, combiner, scalartype
using ITensors.ITensorMPS: MPO
using ITensors.NDTensors: matrix
using LinearAlgebra: diagind, eigen, tr
using Test: @test, @testset

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
    s_matrix(A::Matrix, H::Matrix, ∂H::Matrix)

Construct the action of A (that is, the trace of (∂H+i[A,H])^2) using matrix operations.
"""
function s_matrix(A::Matrix, H::Matrix, ∂H::Matrix)
  G = ∂H + im * (A * H - H * A)
  return real(tr(adjoint(G) * G))
end

"""
    compute_agp(E::Vector, V::Matrix, ∂H::Matrix)

Construct the exact adiabatic gauge potential (AGP).
"""
function compute_agp(E::Vector, V::Matrix, ∂H::Matrix)
  ∂H = adjoint(V) * ∂H * V
  N = length(E)

  Erow = E[reshape(repeat(1:N, N), N, N)]
  AGP = im ./ (transpose(Erow) - Erow)
  AGP = AGP .* ∂H
  AGP[diagind(AGP)] .= zero(Bool)

  return AGP
end

tol(::Type{<:Float32}) = 1e-2
tol(::Type{<:Float64}) = 1e-3
tol(type::Type{<:Complex}) = tol(real(type))
const elts = (Float32, Float64, Complex{Float32}, Complex{Float64})
@testset "ITensorAGP.jl (eltype=$elt)" for elt in elts
  include(joinpath(pkgdir(ITensorAGP), "examples", "transverse_field_ising.jl"))
  res = main(; L=8, hz=0.5, eltype=elt, outputlevel=0)
  AGP = res.AGP
  ls_error = res.ls_error

  @test scalartype(AGP) === complex(elt)
  @test ls_error ≈ 0 atol = tol(elt)

  H = res.H
  dH = res.dH

  ITensors.disable_warn_order()
  H_matrix = matricize(H)
  dH_matrix = matricize(dH)
  AGP_matrix = matricize(AGP)

  E, V = eigen(H_matrix)
  exactAGP = compute_agp(E, V, dH_matrix)
  AGP_matrix = adjoint(V) * AGP_matrix * V  # same basis as `exactAGP`
  AGP_matrix[diagind(AGP_matrix)] .= zero(Bool)  # gauge freedom

  @test s_matrix(AGP_matrix, H_matrix, dH_matrix) ≈ s_matrix(exactAGP, H_matrix, dH_matrix) rtol = tol(
    elt
  )
end
end
