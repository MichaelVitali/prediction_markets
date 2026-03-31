module UtilsFunctions
using Revise
using Statistics
using Distributions
using LinearAlgebra
using DataFrames
using LossFunctions

export  quantile_loss_gradient, initialize_weights, project_to_simplex, quantile_loss

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
