module UtilsFunctions
using Revise
using Statistics
using Distributions
using LinearAlgebra
using DataFrames
using LossFunctions

export  quantile_loss_gradient, initialize_weights, project_to_simplex

    function quantile_loss_gradient(y_true, y_hat, q)
        if y_hat > y_true
            return (1 - q)
        elseif y_true >= y_hat
            return -q
        end
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
