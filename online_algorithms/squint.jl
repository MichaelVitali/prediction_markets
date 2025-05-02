module Squint

using LinearAlgebra
using SpecialFunctions
using QuadGK
include("../functions/functions.jl")
using .UtilsFunctions


export squint, squint_CP

    function squint(forecasters_preds, y_true, T, q)
        n_forecasters = length(keys(forecasters_preds))
        R_t = zeros(n_forecasters)
        V_t = zeros(n_forecasters)

        prior_weights = initialize_weights(n_forecasters)
        learning_rate = 0.01
        weights_history = Matrix{Float64}(undef, n_forecasters, T)
        weights_history[:, 1] = prior_weights

        for t in 2:T
            forecasters_preds_t = [forecasters_preds[f][t] for f in keys(forecasters_preds)]
            agg_quantile_t = sum(forecasters_preds_t .* weights_history[:, t-1])
            lks = quantile_loss_gradient(y_true[t], agg_quantile_t, q) .* forecasters_preds_t

            regrets_t = sum(weights_history[:, t-1] .* lks) .- lks
            R_t += regrets_t
            V_t += lks.^2

            exp_terms = exp.((learning_rate .* R_t) .- (learning_rate.^2 .* V_t)) .* learning_rate
            numerators = prior_weights .* exp_terms
            weights_history[:, t] = numerators ./ sum(numerators)

        end

        return weights_history

    end

    function conjugate_prior_weight(Rk, Vk)
        x = Rk
        y = Vk

        arg1 = x / (2 * sqrt(y))
        arg2 = (x-y) / (2 * sqrt(y))

        if arg1 < (-6) || arg2 > 6
            term1 = (exp(x/2 - y/4) - 1.0) / x
            term2 = (1 - exp(x/2 - y/4)) / (2*y)
        else
            erf_diff = erf(arg1) - erf(arg2)
            term1 = exp(x^2 / (4*y)) * sqrt(pi) * x * erf_diff / (4 * y^(3/2))
            term2 = (1 - exp(x/2 - y/4)) / (2*y)
        end
        return (term1 + term2)
    end


    function squint_CP(forecasters_preds, y_true, T, q)
        n_forecasters = length(keys(forecasters_preds))
        R_t = zeros(n_forecasters)
        V_t = zeros(n_forecasters)

        prior_weights = initialize_weights(n_forecasters)
        weights_history = Matrix{Float64}(undef, n_forecasters, T)
        weights_history[:, 1] = prior_weights

        for t in 2:T
            forecasters_preds_t = [forecasters_preds[f][t] for f in keys(forecasters_preds)]
            agg_quantile_t = sum(forecasters_preds_t .* weights_history[:, t-1])
            lks = quantile_loss_gradient(y_true[t], agg_quantile_t, q) .* forecasters_preds_t

            regrets_t = sum(weights_history[:, t-1] .* lks) .- lks
            R_t += regrets_t
            V_t += lks.^2
            
            exp_terms = [conjugate_prior_weight(R_t[i], V_t[i]) for i in 1:n_forecasters]
            numerators = prior_weights .* exp_terms
            weights_history[:, t] = numerators ./ sum(numerators)
            
        end

        return weights_history

    end

end