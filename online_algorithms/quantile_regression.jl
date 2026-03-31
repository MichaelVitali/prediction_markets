module QuantileRegression

    using LinearAlgebra
    using Statistics

    include("../functions/functions.jl")
    include("../data_generation/DataGeneration.jl")
    using .UtilsFunctions
    using .DataGeneration

export online_quantile_regression, online_quantile_regression_update, online_quantile_regression_update_multiple_lead_times

    function online_quantile_regression(forecasters_preds, forecaster_weights, y_true, T, q)

        forecasters_names = collect(keys(forecasters_preds))
        weights_history = Matrix{Float64}(undef, length(forecaster_weights), T)
        weights_history[:, 1] = forecaster_weights
        learning_rate = 0.01

        for t in 2:T
            forecasters_preds_t = [forecasters_preds[f][t] for f in forecasters_names]
            agg_quantile_t = sum(forecasters_preds_t .* weights_history[:, t-1])
            lks = quantile_loss_gradient(y_true[t], agg_quantile_t, q) .* forecasters_preds_t

            weights_history[:, t] = weights_history[:, t-1] .- learning_rate .* lks
            weights_history[:, t] = project_to_simplex(weights_history[:, t])
            end

            return weights_history
    end

    function online_quantile_regression_update(forecasters_preds, prev_forecaster_weights, y_true, q, learning_rate=0.01)

        agg_quantile_t = sum(forecasters_preds .* prev_forecaster_weights)
        lks = quantile_loss_gradient(y_true, agg_quantile_t, q) .* forecasters_preds

        new_weights = prev_forecaster_weights .- learning_rate .* lks
        new_weights = project_to_simplex(new_weights)

        return new_weights, agg_quantile_t

    end

    function online_quantile_regression_update_multiple_lead_times(forecasters_preds, prev_forecaster_weights, y_true, q, learning_rate=0.01, batch_percentage=0.5)

        n_forecasters = length(forecasters_preds)
        n_lead_times = length(forecasters_preds[1])

        # Kept exactly as requested
        agg_quantile_t = sum(forecasters_preds .* prev_forecaster_weights, dims=1)[1]
        weights = copy(prev_forecaster_weights)

        # Calculate batch dimension based on percentage (minimum size of 1)
        batch_size = max(1, floor(Int, n_lead_times * batch_percentage))

        # Iterate through the data in chunks of batch_size
        for batch_start in 1:batch_size:n_lead_times
            batch_end = min(batch_start + batch_size - 1, n_lead_times)
            current_batch_size = batch_end - batch_start + 1
            
            # Initialize an empty accumulator for the batch gradients
            batch_gradient = zeros(n_forecasters)

            # Accumulate gradients for all points in the current batch
            for t in batch_start:batch_end
                preds_t = [forecasters_preds[i][t] for i in 1:n_forecasters]
                gradient_loss_t = quantile_loss_gradient(y_true[t], agg_quantile_t[t], q)
                
                lks_t = preds_t .* gradient_loss_t
                batch_gradient .+= lks_t
            end

            # Average the gradient over the batch to prevent learning rate inflation
            batch_gradient ./= current_batch_size

            # Update weights and project ONCE per batch
            weights = weights .- learning_rate .* batch_gradient
            weights = project_to_simplex(weights)
        end

        return weights, agg_quantile_t
    end
end
