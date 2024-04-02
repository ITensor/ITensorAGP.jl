# ITensorAGP

This is a package that constructs the adiabatic gauge potential (AGP) as a matrix product operator (MPO), which can be used to adiabatically evolve matrix product states (MPSs). The package is built using the [ITensors.jl](https://github.com/ITensor/ITensors.jl) and [ITensorTDVP.jl](https://github.com/ITensor/ITensorTDVP.jl) libraries.

## Installation

This package is currently not registered. You can install it with:
```julia
julia> using Pkg: Pkg

julia> Pkg.add(url="https://github.com/ITensor/ITensorAGP.jl.git")
```

## Usage

For example, to compute the AGP of the Hamiltonian `H::MPO` with perturbation `dH::MPO`, you can use the `agp` function:
```julia
using ITensorAGP: agp

AGP, ls_error = agp(H, dH; cutoff=1e-6, nsweeps=10, maxdim=40)
```
Given the MPOs `H` and `dH`, this code constructs the AGP as an MPO using the `linsolve` function in [ITensorTDVP.jl](https://github.com/ITensor/ITensorTDVP.jl), which calls the `linsolve` function in [KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl.git).
