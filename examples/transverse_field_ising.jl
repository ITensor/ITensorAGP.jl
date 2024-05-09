using ITensorMPS: MPO, OpSum, siteinds
using ITensorAGP: agp

function ham(s; hz, eltype=Float64)
  L = length(s)

  os = OpSum()
  for j in 1:(L - 1)
    os += 1, "X", j, "X", j + 1
  end
  for j in 1:L
    os += hz, "Z", j
  end

  return MPO(eltype, os, s)
end

function dham(s; eltype=Float64)
  L = length(s)

  os = OpSum()
  for j in 1:L
    os += 1, "Z", j
  end

  return MPO(eltype, os, s)
end

function main(; L, hz, eltype=Float64, outputlevel=1)
  s = siteinds("S=1/2", L)

  H = ham(s; hz, eltype)
  dH = dham(s; eltype)

  agp_hz, ls_error = agp(
    H, dH; l=1, use_real=false, outputlevel, cutoff=1e-6, nsweeps=10, maxdim=40
  )

  return (; agp=agp_hz, ls_error, H, dH)
end
