module Metrics

    using LinearAlgebra
    using Statistics
    using Plots
    using LossFunctions

export calculate_instantaneous_errors, calculate_instantaneous_variance, calculate_instantaneous_quantile_loss

    function calculate_instantaneous_errors(weights_history, target_w)
        n_steps = length(weights_history)

        bias = zeros(n_steps)
        for t in 1:n_steps
            bias[t] = weights_history[t] .- target_w[t]
        end

        return bias
    end

    function calculate_instantaneous_variance(errors, biasses)
        n_steps = length(errors)

        variance = zeros(n_steps)
        for t in 1:n_steps
            variance[t] = (errors[t] - biasses[t])^2
        end

        return variance
    end

    function calculate_instantaneous_quantile_loss(y_hat, y_true, q)

        losses = QuantileLoss(q).(y_hat, y_true)
        return losses
    end

end