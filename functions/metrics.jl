module Metrics

    using LinearAlgebra
    using Statistics
    using Plots
    using LossFunctions

export calculate_bias, calculate_variance, calculate_instantaneous_bias, calculate_instantaneous_quantile_loss, calculate_mean_quantile_loss

    function calculate_bias(weights_history, target_w)
        n_weights, n_steps = size(weights_history)

        bias = zeros(n_weights, n_steps)
        for t in 1:n_steps
            bias[:, t] = mean(weights_history[:, 1:t], dims=2) .- target_w[:, t]
        end
        return bias
    end

    function calculate_instantaneous_bias(weights_history, target_w)
        n_weights, n_steps = size(weights_history)

        bias = zeros(n_weights, n_steps)
        for t in 1:n_steps
            bias[:, t] = weights_history[:, t] .- target_w[:, t]
        end

        return bias
    end

    function calculate_variance(weights_history)
        n_weights, n_steps = size(weights_history)

        variance = zeros(n_weights, n_steps)
        for t in 1:n_steps
            if t == 1
                variance[:, t] .= 0.0
            else
                variance[:, t] = var(weights_history[:, 1:t], dims=2)
            end
        end
        return variance
    end

    function calculate_instantaneous_quantile_loss(forecasts, realizations, q)

        T = length(forecasts)
        losses = QuantileLoss(q).(forecasts, realizations)

        return losses
    end

    function calculate_mean_quantile_loss(forecasts, realizations, q)

        T = length(forecasts)
        mean_loss = zeros(T)
        losses = QuantileLoss(q).(forecasts, realizations)

        for t in 1:T
            mean_loss[t] = mean(losses[1:t])
        end
        
        return mean_loss
    end
end