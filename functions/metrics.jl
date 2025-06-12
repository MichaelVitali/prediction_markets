module Metrics

    using LinearAlgebra
    using Statistics
    using Plots

export calculate_bias, calculate_variance

    function calculate_bias(weights_history, target_w)
        n_weights, n_steps = size(weights_history)

        bias = zeros(n_weights, n_steps)
        for t in 1:n_steps
            bias[:, t] = mean(weights_history[:, 1:t], dims=2) .- target_w[t, :]
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
end