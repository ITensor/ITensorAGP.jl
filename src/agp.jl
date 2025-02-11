using ITensors: δ, addtags, apply, replaceinds, scalartype
using ITensorMPS: MPO, MPS, linsolve, siteinds

"""
    agp(H::MPO, ∂H::MPO, X₀::MPO; use_real = false, init_cutoff = 1e-14, updater_kwargs = (;), kwargs...)
    agp(H::MPO, ∂H::MPO; l = 1, use_real = false, init_cutoff = 1e-14, updater_kwargs = (;), kwargs...)

Construct the adiabatic gauge potential (AGP) as a matrix 
product operator (MPO) by constructing linear equation

    (H^2 \"otimes I + I \"otimes H^2 - 2 H \"otimes H)|x> = |b>

where |x> and |b> = i|∂H H - H ∂H> are matrix product states (MPSs) 
in a doubly large Hilbert space compared to that of H. The AGP as an MPO 
is constructed by solving for the MPS |x> (with initial ansatz |X₀>) 
and converting |X> back into an MPO X.

For H with time reversal symmetry, the AGP and linear equation can be
represented with only real numbers, where solving for X_real in

    (H^2 \"otimes I + I \"otimes H^2 - 2 H \"otimes H)|x_real> = |b_real>

gives X_real = iX (|b_real> = i|b> = -|∂H H - H ∂ H>).

Returns:

    - `X::MPO` - the AGP as an MPO
    - `ls_error::Number` - the `linsolve` error

Optional keyword arguments:

    - `use_real` - boolean specifying whether to use only 
       real numbers/operators (when `H` has time reversal symmetry)
    - `init_cutoff` - float specifying the truncation error cutoff
    - `updater_kwargs` - a `NamedTuple` containing keyword arguments 
       that will get forwarded to the local solver, in this case 
       `KrylovKit.linsolve` which is a GMRES linear solver
"""
function agp(
  H::MPO,
  ∂H::MPO,
  X₀::MPO;
  use_real=false,
  init_cutoff=1e-14,
  updater_kwargs=(;),
  outputlevel=0,
  kwargs...,
)
  # solver kwargs
  default_updater_kwargs = (; ishermitian=true, rtol=1e-4, maxiter=1, krylovdim=3)

  # user input `updater_kwargs` will override `default_updater_kwargs` if duplicates exist
  updater_kwargs = (; default_updater_kwargs..., updater_kwargs...)

  # length of MPO
  L = length(H)

  # construct the sites
  s = [siteinds(H)[j][2] for j in 1:L]

  # construct identity MPO
  I_mpo = MPO([δ(scalartype(H), dag(s[j])', s[j]) for j in 1:L])

  # convert X₀ from MPO to MPS
  X₀ = mpo_to_mps(X₀)
  X₀ = MPS([
    isodd(j) ? X₀[j] : replaceinds(X₀[j], s[Int(j / 2)]' => addtags(s[Int(j / 2)], "ket"))
    for j in 1:(2L)
  ])

  H_bra = interleave(H, addtags(I_mpo, "ket"))
  H_ket = interleave(I_mpo, addtags(H, "ket"))

  H1 = apply(H_bra, H_bra; cutoff=init_cutoff)
  H2 = apply(H_ket, H_ket; cutoff=init_cutoff)
  H3 = apply(H_bra, H_ket; cutoff=init_cutoff)

  # construct A
  A = +(H1, H2, -2 * H3; cutoff=init_cutoff)

  # construct b
  b = -(
    apply(∂H, H; cutoff=init_cutoff), apply(H, ∂H; cutoff=init_cutoff); cutoff=init_cutoff
  )
  b = use_real ? -b : im * b

  # convert b from MPO to MPS
  b = mpo_to_mps(b)
  b = MPS([
    isodd(j) ? b[j] : replaceinds(b[j], s[Int(j / 2)]' => addtags(s[Int(j / 2)], "ket")) for
    j in 1:(2L)
  ])

  # solve linear equation Ax = b
  outputlevel > 0 && @show linsolve_error(A, b, X₀)
  linsolve_time = @elapsed begin
    X = linsolve(A, b, X₀; updater_kwargs, kwargs...)
  end
  outputlevel > 0 && @show linsolve_time

  linsolve_error_AXb = linsolve_error(A, b, X)
  outputlevel > 0 && @show linsolve_error_AXb

  # convert X from MPS to MPO
  X = MPO([X[j] * X[j + 1] for j in 1:2:(2L)])
  X = MPO([replaceinds(X[j], addtags(s[j], "ket") => s[j]') for j in 1:L])

  return X, linsolve_error_AXb
end

function agp(
  H::MPO, ∂H::MPO; l=1, use_real=false, init_cutoff=1e-14, updater_kwargs=(;), kwargs...
)
  X₀ = find_ansatz(H, ∂H, l; use_real, init_cutoff, kwargs...)
  return agp(H, ∂H, X₀; use_real, init_cutoff, updater_kwargs, kwargs...)
end
