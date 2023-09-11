using ITensors

# |Ax - b| = √((⟨x|A† - ⟨b|)(A|x⟩ - |b⟩))
#          = √(⟨x|A†A|x⟩ + ⟨b|b⟩ - 2 * real(⟨b|A|x⟩))
# normalized
function linsolve_error(A, b, x)
    return √(
        real((inner(A, x, A, x) + inner(b, b) - 2 * real(inner(b', A, x)))) /
        real(inner(b, b)),
    )
end

function find_ansatz(
    H::MPO,
    ∂H::MPO,
    l::Integer;
    use_real = false,
    init_cutoff = 1e-14,
    kwargs...,
)
    cutoff = init_cutoff

    adH = Array{MPO}(undef, 2l)
    M = zeros(Float64, (l, l))
    adH[1] = -(apply(H, ∂H; cutoff), apply(∂H, H; cutoff); cutoff)
    for k = 2:2l
        Hc = adH[k-1]
        adH[k] = -(apply(H, Hc; cutoff), apply(Hc, H; cutoff); cutoff)
    end

    ba = [-(inner(∂H, adH[2*m]) + inner(adH[2*m], ∂H)) for m = 1:l]
    for m = 1:l
        M[m, :] = [inner(adH[2*m], adH[2*k]) + inner(adH[2*k], adH[2*m]) for k = 1:l]
    end

    α = M \ ba

    X₀ = α[1] * adH[1]
    for k = 2:l
        X₀ = +(X0, α[k] * adH[2*k-1]; cutoff)
    end

    if !use_real
        X₀ *= im
    else
        X₀ *= -1.0
    end

    return X₀
end
