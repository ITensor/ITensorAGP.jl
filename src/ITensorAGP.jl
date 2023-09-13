module ITensorAGP

using ITensors
using ITensorTDVP

include("agp_utils.jl")
include("mps_utils.jl")

"""
    agp(H::MPO, ∂H::MPO, X₀::MPO; use_real = false, init_cutoff = 1e-14, solver_kwargs = (;), kwargs...)
    agp(H::MPO, ∂H::MPO; l = 1, use_real = false, init_cutoff = 1e-14, solver_kwargs = (;), kwargs...)

Construct the adiabatic gauge potential (AGP) as a matrix 
product operator (MPO) by constructing linear equation

    (H^2 \otimes I + I \otimes H^2 - 2 H \otimes H)|x> = |b>

where |x> and |b> = i|∂H H - H ∂H> are matrix product states (MPSs) 
in a doubly large Hilbert space compared to that of H. The AGP as an MPO 
is constructed by solving for the MPS |x> (with initial ansatz |X₀>) 
and converting |X> back into an MPO X.

For H with time reversal symmetry, the AGP and linear equation can be
represented with only real numbers, where solving for X_real in

    (H^2 \otimes I + I \otimes H^2 - 2 H \otimes H)|x_real> = |b_real>

gives X_real = iX (|b_real> = i|b> = -|∂H H - H ∂ H>).

Returns:

    - `X::MPO` - the AGP as an MPO
    - `ls_error::Number` - the `linsolve` error

Optional keyword arguments:

    - `use_real` - boolean specifying whether to use only 
       real numbers/operators (when `H` has time reversal symmetry)
    - `init_cutoff` - float specifying the truncation error cutoff
    - `solver_kwargs` - a `NamedTuple` containing keyword arguments 
       that will get forwarded to the local solver, in this case 
       `KrylovKit.linsolve` which is a GMRES linear solver
"""
function agp(
  H::MPO, ∂H::MPO, X₀::MPO; use_real=false, init_cutoff=1e-14, solver_kwargs=(;), kwargs...
)
  #   outputlevel = get(kwargs, :outputlevel, 0)
  #   cutoff = get(kwargs, :cutoff, 1e-8)
  #   nsweeps = get(kwargs, :nsweeps, 10)
  #   maxdim = get(kwargs, :maxdim, 40)

  #   # solver kwargs
  #   linsolve_ishermitian = get(solver_kwargs, :ishermitian, true)
  #   linsolve_rtol = get(solver_kwargs, :rtol, 1e-4)
  #   linsolve_maxiter = get(solver_kwargs, :maxiter, 1)
  #   linsolve_krylovdim = get(solver_kwargs, :krylovdim, 3)

  #   linsolve_kwargs = (;
  #     ishermitian=linsolve_ishermitian,
  #     rtol=linsolve_rtol,
  #     maxiter=linsolve_maxiter,
  #     krylovdim=linsolve_krylovdim,
  #   )

  # length of MPO
  L = length(H)

  # construct the sites
  s = [siteinds(H)[j][2] for j in 1:L]

  # construct identity MPO
  I_mpo = MPO([δ(dag(s[j])', s[j]) for j in 1:L])

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
  if !use_real
    b =
      im * (-(
        apply(∂H, H; cutoff=init_cutoff),
        apply(H, ∂H; cutoff=init_cutoff);
        cutoff=init_cutoff,
      ))
  else
    b = -(-(
      apply(∂H, H; cutoff=init_cutoff),
      apply(H, ∂H; cutoff=init_cutoff);
      cutoff=init_cutoff,
    ))
  end

  # convert b from MPO to MPS
  b = mpo_to_mps(b)
  b = MPS([
    isodd(j) ? b[j] : replaceinds(b[j], s[Int(j / 2)]' => addtags(s[Int(j / 2)], "ket")) for
    j in 1:(2L)
  ])

  # solve linear equation Ax = b
  if outputlevel > 0
    @show linsolve_error(A, b, X₀)
    X = @time linsolve(A, b, X₀; solver_kwargs_kwargs, kwargs...)
  else
    X = linsolve(A, b, X₀; solver_kwargs_kwargs, kwargs...)
  end

  ls_error = linsolve_error(A, b, X)

  if outputlevel > 0
    @show ls_error
  end

  # convert X from MPS to MPO
  X = MPO([X[j] * X[j + 1] for j in 1:2:(2L)])
  X = MPO([replaceinds(X[j], addtags(s[j], "ket") => s[j]') for j in 1:L])

  return X, ls_error
end

function agp(
  H::MPO, ∂H::MPO; l=1, use_real=false, init_cutoff=1e-14, solver_kwargs=(;), kwargs...
)
  X₀ = find_ansatz(H, ∂H, l; use_real, init_cutoff, kwargs...)
  return agp(H, ∂H, X₀; use_real, init_cutoff, solver_kwargs, kwargs...)
end

export agp

end
