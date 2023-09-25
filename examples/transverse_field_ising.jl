using ITensors
using ITensorAGP

function ham(s; hz)
  L = length(s)

  os = OpSum()
  for j in 1:(L - 1)
    os += 1.0, "X", j, "X", j + 1
  end
  for j in 1:L
    os += hz, "Z", j
  end

  return MPO(os, s)
end

function dham(s)
  L = length(s)

  os = OpSum()
  for j in 1:L
    os += 1.0, "Z", j
  end

  return MPO(os, s)
end

function main(; L, hz, outputlevel=1)
  s = siteinds("S=1/2", L)

  H = ham(s; hz)
  dH = dham(s)

  AGP, ls_error = agp(
    H, dH; l=1, use_real=false, outputlevel, cutoff=1e-6, nsweeps=10, maxdim=40
  )

  return (; AGP, ls_error, H, dH)
end
