module UtilsFunctions
using Revise
using Statistics
using Distributions
using LinearAlgebra
using DataFrames
using LossFunctions

export  quantile_loss_gradient, initialize_weights, project_to_simplex, quantile_loss, padded_ylims

    """
        padded_ylims(arrays...; pad_frac=0.05)

    Return a shared `(lo, hi)` y-axis range covering every finite value in the
    given arrays (vectors or matrices), expanded by `pad_frac` of the span on
    each side. Use it to give all panels of a figure (and comparable figures)
    identical y-limits so they are directly comparable.
    """
    function padded_ylims(arrays...; pad_frac=0.05)
        vals = Float64[]
        for a in arrays
            append!(vals, vec(Float64.(collect(a))))
        end
        filter!(isfinite, vals)
        isempty(vals) && return (0.0, 1.0)
        lo, hi = minimum(vals), maximum(vals)
        if hi == lo
            pad = hi == 0 ? 1.0 : abs(hi) * pad_frac
            return (lo - pad, hi + pad)
        end
        pad = (hi - lo) * pad_frac
        return (lo - pad, hi + pad)
    end

    function quantile_loss(y_true, y_hat, q)
        error = y_true - y_hat
        return error > 0 ? q * error : (q - 1) * error
    end

    function quantile_loss_gradient(y_true, y_hat, q)
        return y_hat > y_true ? (1 - q) : -q
    end

    
    function initialize_weights(n_experts::Integer)

        weigths = fill(1 / n_experts, n_experts)
        return weigths
    
    end

    function project_to_simplex(v)
        n = length(v)
        u = sort(v, rev=true)
        cssv = cumsum(u) .- 1

        rho = findlast(k -> u[k] > cssv[k] / k, 1:n)
        tau = cssv[rho] / rho
        w = max.(v .- tau, 0.0)

        return w
    end

end
