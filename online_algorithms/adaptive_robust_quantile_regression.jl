module AdaptiveRobustRegression

using LinearAlgebra

include("../functions/functions.jl")
using .UtilsFunctions

export online_adaptive_robust_quantile_regression

    function online_adaptive_robust_quantile_regression(x, y, prev_w, prev_D, alpha, q, learning_rate=0.01)

        masked_x = x .* (1 .- alpha)
        agg_quantile_t = sum((prev_w .+ prev_D * alpha) .* masked_x)

        gradient_w = quantile_loss_gradient(y, agg_quantile_t, q) .* masked_x
        gradient_D = quantile_loss_gradient(y, agg_quantile_t, q) .* masked_x * alpha'

        new_w = prev_w .- learning_rate .* gradient_w
        new_w = project_to_simplex(new_w)
        new_D = prev_D - learning_rate .* gradient_D

        return new_w, new_D

    end

end