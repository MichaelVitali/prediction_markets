module RobustOptimizationBenchmarks

using LinearAlgebra
using Statistics

include("../functions/functions.jl")
include("quantile_regression.jl")
using .UtilsFunctions
using .QuantileRegression

export quantile_regression_mean_imputation, quantile_regression_last_impute, quantile_regression_last_impute_multiple_lead_times, quantile_regression_mean_imputation_multiple_lead_times

    function quantile_regression_mean_imputation(forecasters_preds, y, w, alpha, q, learning_rate=0.01)

        n_forecasters, T = size(forecasters_preds)
        received_f = Dict([i => copy(forecasters_preds[i, 1:100]) for i in 1:n_forecasters])
        forecasts_history = zeros(T)
        weights_history = zeros(n_forecasters, T)
        for t in 1:500
            weights_history[:, t] = w
            forecasts_history[t] = 0.0
        end

        for t in 500:T
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

    """
        quantile_regression_mean_imputation_multiple_lead_times(forecasters_preds, y, w, alpha, q, learning_rate, batch_percentage)

    Mean-imputation benchmark: a missing forecast (`alpha[i, t] == 1`) is replaced by the mean (per lead time)
    of the forecasts received from that seller so far, then the QR step (`qr_aggregate` + `qr_update`, projected
    on the simplex) is applied. The first session's forecast seeds the history. With `alpha == 0` it is exactly QR.
    """
    function quantile_regression_mean_imputation_multiple_lead_times(forecasters_preds, y, w, alpha, q, learning_rate=0.01, batch_percentage=0.5)

        n_forecasters, T = size(forecasters_preds)
        n = length(forecasters_preds[1, 1]) # lead time
        received_sum = [copy(forecasters_preds[i, 1]) for i in 1:n_forecasters]   # running sum of received forecasts
        received_count = ones(Int, n_forecasters)
        forecasts_history = zeros(T, n)
        weights_history = zeros(n_forecasters, T)
        weights_history[:, 1] = w

        for t in 2:T
            # Mean imputation
            forecasters_preds_t = Vector{Vector{Float64}}(undef, n_forecasters)
            for i in 1:n_forecasters
                if alpha[i, t] == 1
                    forecasters_preds_t[i] = received_sum[i] ./ received_count[i]
                else
                    forecasters_preds_t[i] = forecasters_preds[i, t]
                    received_sum[i] .+= forecasters_preds[i, t]
                    received_count[i] += 1
                end
            end

            # Same combination and projected step as QR
            combined_quantile = qr_aggregate(forecasters_preds_t, weights_history[:, t-1])
            weights_history[:, t] = qr_update(forecasters_preds_t, weights_history[:, t-1], y[t], combined_quantile, q, learning_rate, batch_percentage)
            forecasts_history[t, :] .= combined_quantile
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

    """
        quantile_regression_last_impute_multiple_lead_times(forecasters_preds, y, w, alpha, q, learning_rate, batch_percentage)

    Last-value-imputation benchmark: a missing forecast (`alpha[i, t] == 1`) is replaced by the last forecast
    received from that seller, then the QR step (`qr_aggregate` + `qr_update`, projected on the simplex) is
    applied. The first session's forecast seeds the history. With `alpha == 0` it is exactly QR.
    """
    function quantile_regression_last_impute_multiple_lead_times(forecasters_preds, y, w, alpha, q, learning_rate=0.01, batch_percentage=0.5)

        n_forecasters, T = size(forecasters_preds)
        n = length(forecasters_preds[1, 1])
        last_forecasts = [forecasters_preds[i, 1] for i in 1:n_forecasters]
        forecasts_history = zeros(T, n)
        weights_history = zeros(n_forecasters, T)
        weights_history[:, 1] = w

        for t in 2:T
            # Last forecast imputation
            for i in 1:n_forecasters
                if alpha[i, t] == 0
                    last_forecasts[i] = forecasters_preds[i, t]
                end
            end
            forecasters_preds_t = copy(last_forecasts)

            # Same combination and projected step as QR
            combined_quantile = qr_aggregate(forecasters_preds_t, weights_history[:, t-1])
            weights_history[:, t] = qr_update(forecasters_preds_t, weights_history[:, t-1], y[t], combined_quantile, q, learning_rate, batch_percentage)
            forecasts_history[t, :] .= combined_quantile
        end

        return weights_history, forecasts_history
    end
end