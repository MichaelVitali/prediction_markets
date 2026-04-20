module AdaptiveRobustRegression

using LinearAlgebra
using Statistics

include("../functions/functions.jl")
using .UtilsFunctions

export online_adaptive_robust_quantile_regression, online_adaptive_robust_quantile_regression_multiple_lead_times, online_adaptive_robust_quantile_regression_multiple_lead_times_trial

    function online_adaptive_robust_quantile_regression(x, y, prev_w, prev_D, alpha, q, learning_rate=0.01)

        """
            Function calculates the update step for the adaptive robust quantile regression method. This function works only for lead time = 1.
        """

        masked_x = x .* (1 .- alpha)
        agg_quantile_t = sum((prev_w .+ prev_D * alpha) .* masked_x)

        gradient_w = quantile_loss_gradient(y, agg_quantile_t, q) .* masked_x
        gradient_D = quantile_loss_gradient(y, agg_quantile_t, q) .* masked_x * alpha'

        new_w = prev_w .- learning_rate .* gradient_w
        new_w = project_to_simplex(new_w)
        new_D = prev_D - learning_rate .* gradient_D

        return new_w, new_D, agg_quantile_t

    end

    function online_adaptive_robust_quantile_regression_multiple_lead_times(x, y, prev_w, prev_D, alpha, q, learning_rate=0.01, batch_percentage=0.5)

        """
            Function calculates the update step for the adaptive robust quantile regression method. This function works for multiple lead times.
        """

        n_forecasters = length(x)
        n_lead_times = length(x[1])
        
        masked_x = x .* (1 .- alpha)
        effective_w = prev_w .+ (prev_D * alpha)
        effective_w = project_to_simplex(effective_w)
        
        agg_quantile_t = sum(masked_x .* effective_w)
        weights = copy(prev_w)
        D = copy(prev_D)

        batch_size = max(1, floor(Int, n_lead_times * batch_percentage))

        # Iterate through the data in chunks of batch_size
        for batch_start in 1:batch_size:n_lead_times
            batch_end = min(batch_start + batch_size - 1, n_lead_times)
            current_batch_size = batch_end - batch_start + 1
            
            # Initialize empty accumulators for both weights and the D matrix
            batch_grad_w = zeros(n_forecasters)
            batch_grad_D = zeros(size(D)) 
            
            # Accumulate gradients for all points in the current batch
            for t in batch_start:batch_end
                preds_t = [masked_x[i][t] for i in 1:n_forecasters]
                gradient_loss_t = quantile_loss_gradient(y[t], agg_quantile_t[t], q)
                
                # Calculate individual gradients
                grad_w_t = preds_t .* gradient_loss_t
                grad_D_t = grad_w_t * alpha'
                
                # Add to batch accumulators
                batch_grad_w .+= grad_w_t
                batch_grad_D .+= grad_D_t
            end
            
            # Average the gradients over the batch to maintain a stable learning rate
            batch_grad_w ./= current_batch_size
            batch_grad_D ./= current_batch_size
            
            # Update weights and matrix ONCE per batch
            weights = weights .- learning_rate .* batch_grad_w
            weights = project_to_simplex(weights)
            D = D .- learning_rate .* batch_grad_D
        end

        return weights, D, agg_quantile_t
    end

    function online_adaptive_robust_quantile_regression_multiple_lead_times_trial(x, y, prev_w, prev_D, alpha, q, learning_rate=0.01, batch_percentage=0.5)

        """
            Function calculates the update step for the adaptive robust quantile regression method. This function works for multiple lead times.
        """

        n_forecasters = length(x)
        n_lead_times = length(x[1])
        
        effective_w = prev_w .+ (prev_D * alpha)
        effective_w = effective_w .* (1 .- alpha)
        available = alpha .< 1
        projected = project_to_simplex(effective_w[available])
        effective_w = zeros(length(effective_w))
        effective_w[available] = projected

        masked_x = x .* (1 .- alpha)
        agg_quantile_t = sum(masked_x .* effective_w)
        weights = copy(prev_w)
        D = copy(prev_D)

        batch_size = max(1, floor(Int, n_lead_times * batch_percentage))

        # Iterate through the data in chunks of batch_size
        for batch_start in 1:batch_size:n_lead_times
            batch_end = min(batch_start + batch_size - 1, n_lead_times)
            current_batch_size = batch_end - batch_start + 1
            
            # Initialize empty accumulators for both weights and the D matrix
            batch_grad_w = zeros(n_forecasters)
            batch_grad_D = zeros(size(D)) 
            
            # Accumulate gradients for all points in the current batch
            for t in batch_start:batch_end
                preds_t = [masked_x[i][t] for i in 1:n_forecasters]
                gradient_loss_t = quantile_loss_gradient(y[t], agg_quantile_t[t], q)
                
                # Calculate individual gradients
                grad_w_t = (1 .- alpha) .* preds_t .* gradient_loss_t
                grad_D_t = grad_w_t * alpha'
                
                # Add to batch accumulators
                batch_grad_w .+= grad_w_t
                batch_grad_D .+= grad_D_t
            end
            
            # Average the gradients over the batch to maintain a stable learning rate
            batch_grad_w ./= current_batch_size
            batch_grad_D ./= current_batch_size
            
            # Update weights and matrix ONCE per batch
            weights = weights .- learning_rate .* batch_grad_w
            weights = project_to_simplex(weights)
            D = D .- learning_rate .* batch_grad_D
        end

        return weights, D, agg_quantile_t
    end

end