module AdaptiveRobustRegression

using LinearAlgebra
using Statistics

include("../functions/functions.jl")
using .UtilsFunctions

export online_adaptive_robust_quantile_regression, online_adaptive_robust_quantile_regression_multiple_lead_times, rqr_aggregate, rqr_update

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

    # Previous implementation (raw gradient step on w and D), superseded by the projected update below
    #=
    function online_adaptive_robust_quantile_regression_multiple_lead_times(x, y, prev_w, prev_D, alpha, q, learning_rate=0.01, batch_percentage=0.5)

        """
            Function calculates the update step for the adaptive robust quantile regression method. This function works for multiple lead times.
        """

        n_forecasters = length(x)
        n_lead_times = length(x[1])
        
        masked_x = x .* (1 .- alpha)
        effective_w = (prev_w .+ (prev_D * alpha)) .* (1 .- alpha)
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
    =#

    """
        online_adaptive_robust_quantile_regression_multiple_lead_times(x, y, prev_w, prev_D, alpha, q, learning_rate, batch_percentage)

    Same affine model as the previous implementation above (effective weights `w + D*alpha` on the sub-simplex of the
    available forecasters), but the update is the *projected* step of the effective weights:

        w_eff_new = project_to_simplex(w_eff - lr * grad)      (available forecasters only)
        delta     = w_eff_new - w_eff
        w        += delta,   D += delta * alpha'

    Both `w` and `D` move by the displacement the constrained problem actually takes. When the
    reduced-ensemble optimum sits on a corner of the simplex (target outside the hull of the available
    forecasters) the raw gradient never vanishes, but the projected step does, so neither `w` nor `D`
    is pushed indefinitely. `delta` sums to zero over the available forecasters, so the base weight of
    an absent forecaster is never drained through the projection. With `alpha == 0` this reduces exactly
    to the QR update.
    """
    function online_adaptive_robust_quantile_regression_multiple_lead_times(x, y, prev_w, prev_D, alpha, q, learning_rate=0.01, batch_percentage=0.5)

        agg_quantile_t = rqr_aggregate(x, prev_w, prev_D, alpha)
        weights, D = rqr_update(x, y, prev_w, prev_D, alpha, agg_quantile_t, q, learning_rate, batch_percentage)

        return weights, D, agg_quantile_t
    end

    # Effective weights: base + correction, restricted to the available forecasters, on their sub-simplex
    function effective_weights(w, D, alpha)
        available = alpha .< 1
        e = (w .+ D * alpha) .* (1 .- alpha)
        out = zeros(length(w))
        out[available] = project_to_simplex(e[available])
        return out
    end

    """
        rqr_aggregate(x, prev_w, prev_D, alpha)

    Combined forecast (one value per lead time) issued in the session, using the effective weights
    of the available forecasters (`alpha[i] == 1` marks forecaster `i` as missing).
    """
    function rqr_aggregate(x, prev_w, prev_D, alpha)
        masked_x = x .* (1 .- alpha)
        w_eff = effective_weights(prev_w, prev_D, alpha)
        return sum(masked_x .* w_eff)
    end

    """
        rqr_update(x, y, prev_w, prev_D, alpha, agg_quantile_t, q, learning_rate, batch_percentage)

    Projected update of `w` and `D` once the realization `y` is observed. The gradient is evaluated
    at the forecast `agg_quantile_t` issued in the session (frozen over the batches).
    """
    function rqr_update(x, y, prev_w, prev_D, alpha, agg_quantile_t, q, learning_rate=0.01, batch_percentage=0.5)

        n_forecasters = length(x)
        n_lead_times = length(x[1])
        available = alpha .< 1
        masked_x = x .* (1 .- alpha)

        weights = copy(prev_w)
        D = copy(prev_D)
        w_eff = effective_weights(weights, D, alpha)

        batch_size = max(1, floor(Int, n_lead_times * batch_percentage))

        for batch_start in 1:batch_size:n_lead_times
            batch_end = min(batch_start + batch_size - 1, n_lead_times)
            current_batch_size = batch_end - batch_start + 1

            batch_grad = zeros(n_forecasters)
            for t in batch_start:batch_end
                preds_t = [masked_x[i][t] for i in 1:n_forecasters]
                batch_grad .+= preds_t .* quantile_loss_gradient(y[t], agg_quantile_t[t], q)
            end
            batch_grad ./= current_batch_size

            # Projected step on the effective weights of the available forecasters
            w_eff_new = zeros(n_forecasters)
            w_eff_new[available] = project_to_simplex(w_eff[available] .- learning_rate .* batch_grad[available])
            delta = w_eff_new .- w_eff

            # Distribute the displacement to the base weights and the correction column(s)
            weights = project_to_simplex(weights .+ delta)
            D = D .+ delta * alpha'
            w_eff = effective_weights(weights, D, alpha)
        end

        return weights, D
    end

end