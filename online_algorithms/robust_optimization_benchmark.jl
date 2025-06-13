module RobustOptimizationBenchmarks

using LinearAlgebra
using Statistics

include("../functions/functions.jl")
using .UtilsFunctions

export quantile_regression_mean_imputation, quantile_regression_last_impute, quantile_regression_oracle

    function quantile_regression_mean_imputation(forecasters_preds, y, w, alpha, q, learning_rate=0.01)

        n_forecasters, T = size(forecasters_preds)
        received_f = Dict([i => copy(forecasters_preds[i, 1:100]) for i in 1:n_forecasters])
        forecasts_history = zeros(T)
        weights_history = zeros(n_forecasters, T)
        for t in 1:100
            weights_history[:, t] = w
            forecasts_history[t] = 0.0
        end

        for t in 100:T
            forecasters_preds_t = zeros(3)
            prev_w = weights_history[:, t-1]
            y_t = y[t]

            # Mean imputation
            for i in 1:n_forecasters
                if alpha[i, t] == 1
                    forecasters_preds_t[i] = mean(received_f[i])
                else
                    forecasters_preds_t[i] = forecasters_preds[i, t]
                    push!(received_f[i], forecasters_preds[i, t])
                end
            end

            # Combination and gradient step
            combined_quantile = sum(forecasters_preds_t .* prev_w)
            gradient_w = quantile_loss_gradient(y_t, combined_quantile, q) .* forecasters_preds_t

            weights_history[:, t] = prev_w .- learning_rate .* gradient_w
            forecasts_history[t] = combined_quantile
        end

        return weights_history, forecasts_history
    end

    function quantile_regression_last_impute(forecasters_preds, y, w, alpha, q, learning_rate=0.01)

        n_forecasters, T = size(forecasters_preds)
        last_forecasts = Dict([i => forecasters_preds[i, 1] for i in 1:n_forecasters])
        forecasts_history = zeros(T)
        weights_history = zeros(n_forecasters, T)
        weights_history[:, 1] = w

        for t in 2:T
            forecasters_preds_t = zeros(3)
            prev_w = weights_history[:, t-1]
            y_t = y[t]

            # Last forecast imputation
            for i in 1:n_forecasters
                if alpha[i, t] == 1
                    forecasters_preds_t[i] = last_forecasts[i]
                else
                    forecasters_preds_t[i] = forecasters_preds[i, t]
                    last_forecasts[i] = forecasters_preds[i, t]
                end
            end

            # Combination and gradient step
            combined_quantile = sum(forecasters_preds_t .* prev_w)
            gradient_w = quantile_loss_gradient(y_t, combined_quantile, q) .* forecasters_preds_t

            weights_history[:, t] = prev_w .- learning_rate .* gradient_w
            forecasts_history[t] = combined_quantile
        end

        return weights_history, forecasts_history
    end

    function quantile_regression_oracle(forecasters_preds, y, w, alpha, q, learning_rate=0.01)

        n_forecasters, T = size(forecasters_preds)
        weights_history = zeros(n_forecasters, T)
        forecasts_history = zeros(T)
        weights_history[:, 1] = w

        for t in 2:T
            prev_w = weights_history[:, t-1]
            y_t = y[t]

            forecasters_preds_t = forecasters_preds[:, t] .* (1 .- alpha[:, t])

            # Combination and gradient step
            combined_quantile = sum(forecasters_preds_t .* prev_w)

            if sum(alpha[:, t]) == 0
                gradient_w = quantile_loss_gradient(y_t, combined_quantile, q) .* forecasters_preds_t
            else
                gradient_w = 0
            end

            weights_history[:, t] = prev_w .- learning_rate .* gradient_w
            forecasts_history[t] = combined_quantile
        end

        return weights_history, forecasts_history
    end
end