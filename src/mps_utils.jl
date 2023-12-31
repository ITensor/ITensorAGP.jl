normalized_inner(x, y) = inner(x, y) / (norm(x) * norm(y))

# TODO: Move to ITensors.jl
using ITensors: AbstractMPS

function Base.vcat(ψ₁::MPST, ψ₂::MPST) where {MPST<:AbstractMPS}
  return MPST([ITensors.data(ψ₁); ITensors.data(ψ₂)])
end
Base.vcat(a::ITensor, ψ::AbstractMPS) = [typeof(ψ)([a]); ψ]
Base.vcat(a::Vector{ITensor}, ψ::AbstractMPS) = [typeof(ψ)(a); ψ]
Base.vcat(ψ::AbstractMPS, a::ITensor) = [ψ; typeof(ψ)([a])]
Base.vcat(ψ::AbstractMPS, a::Vector{ITensor}) = [ψ; typeof(ψ)(a)]
Base.reverse(ψ::AbstractMPS) = typeof(ψ)(reverse(ITensors.data(ψ)))
Base.pop!(ψ::AbstractMPS) = pop!(ITensors.data(ψ))

#
# Take an MPS/MPO of length `n`:
#
# A₁-A₂-A₃-A₄-…-Aₙ
#
# and turn it into an MPS/MPO of length `2n - 1`:
#
# A₁-δ-A₂-δ-A₃-δ-A₄-δ-…-δ-Aₙ
#
# where the new even sites are `δ` tensors.
#
function insert_identity_links(ψ::AbstractMPS)
  ψ = insert_missing_links(ψ)
  n = length(ψ)
  ψ̃ = typeof(ψ)(2n - 1)
  for j in 1:n
    ψ̃[2j - 1] = ψ[j]
  end
  l = linkinds(ψ)
  l̃ = sim(l)
  δ⃗ = [δ(dag(l[j]), l̃[j]) for j in 1:(n - 1)]
  for j in 1:(n - 1)
    ψ̃[2j] = δ⃗[j]
    ψ̃[2j + 1] *= dag(δ⃗[j])
  end
  return ψ̃
end

function combine_linkinds(ψ::AbstractMPS)
  n = length(ψ)
  ψ̃ = copy(ψ)
  for j in 1:(n - 1)
    Cj = combiner(commoninds(ψ̃[j], ψ̃[j + 1]))
    ψ̃[j] = ψ̃[j] * Cj
    ψ̃[j + 1] = ψ̃[j + 1] * dag(Cj)
  end
  return ψ̃
end

# TODO: generalize to more arguments
function interleave(ψ₁::AbstractMPS, ψ₂::AbstractMPS)
  @assert length(ψ₁) == length(ψ₂)
  n = length(ψ₁)
  @assert typeof(ψ₁) == typeof(ψ₂)
  ψ̃₁ = [insert_identity_links(ψ₁); ITensor(1.0)]
  ψ̃₂ = [ITensor(1.0); insert_identity_links(ψ₂)]
  ψ̃ = typeof(ψ₁)(2n)
  for j in 1:(2n)
    ψ̃[j] = ψ̃₁[j] * ψ̃₂[j]
  end
  return combine_linkinds(ψ̃)
end
function interleave(ψ₁::AbstractMPS, ψ₂::AbstractMPS, ψ₃::AbstractMPS, ψ::AbstractMPS...)
  return error("Not implemented")
end
interleave(ψ::AbstractMPS) = ψ

function interleave(x::Vector{T}, y::Vector{T}) where {T}
  @assert length(x) == length(y)
  n = length(x)
  z = Vector{T}(undef, 2n)
  for j in 1:n
    z[2j - 1] = x[j]
    z[2j] = y[j]
  end
  return z
end
interleave(x::Vector) = x

function siteinds_per_dimension(::Val{ndims}, ψ::MPS) where {ndims}
  s = siteinds(ψ)
  return ntuple(j -> s[j:ndims:end], Val(ndims))
end

function insert_missing_links(ψ::AbstractMPS)
  ψ = copy(ψ)
  n = length(ψ)
  for j in 1:(n - 1)
    if !hascommoninds(ψ[j], ψ[j + 1])
      lⱼ = Index(1, "j=$(j)↔$(j + 1)")
      ψ[j] *= onehot(lⱼ => 1)
      ψ[j + 1] *= onehot(dag(lⱼ) => 1)
    end
  end
  return ψ
end

"""
    mpo_to_mps(A::MPO; cutoff=1e-15)

Convert an MPO of length `n` with pairs of primed
and unprimed indices into an MPS of length `2n`
where odd sites have the unprimed indices 
of the MPO and even sites have the primed indices
of the MPO.
"""
function mpo_to_mps(A::MPO; cutoff=1e-15)
  n = length(A)
  A_mps = MPS(2n)
  for j in 1:n
    A_mps[2j - 1], A_mps[2j] = factorize(
      A[j], [linkinds(A, j - 1); filterinds(siteinds(A, j); plev=0)]; cutoff
    )
  end
  return A_mps
end
