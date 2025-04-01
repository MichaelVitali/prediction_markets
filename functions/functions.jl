module UtilsFunctions
using Revise
using Statistics
using Distributions
using LinearAlgebra
using DataFrames
using LossFunctions

export  get_avg_distribution, quantile_averaging_dist_multiple_times, quantile_averaging_dists, quantile_averaging_dataframes

    function quantile_averaging_dists(dists, quantiles, weights)

        generated_quantiles = Matrix{Float64}(undef, length(dists), length(quantiles))
        for (i, dist) in enumerate(dists)
            dist_quantiles = [quantile(dist, quant) for quant in quantiles]
            generated_quantiles[i, :] = dist_quantiles
        end

        weights = weights / sum(weights)
        averaged_quantiles = generated_quantiles .* weights
        averaged_quantiles = sum(averaged_quantiles, dims=1)

        return averaged_quantiles
    end

    function quantile_averaging_dist_multiple_times(forecasters_dists, quantiles, weigths, T)

        avg_quantiles = Matrix{Float64}(undef, length(quantiles), T)
        for t in range(1, stop=T)
            dists_t = values(forecasters_dists[t])
            avg = quantile_averaging_dists(dists_t, quantiles, weigths)
            avg_quantiles[:, t] = avg
        end

        return avg_quantiles
    end

    function quantile_averaging_dataframes(forecasters_dfs::Dict, quantiles, weights, T)

        q_names = ["q"*string(Int(q*100)) for q in quantiles]
        avg_quantiles_df = DataFrame([name => zeros(T) for name in q_names])
    
        #weights = weights / sum(weights)
        for q in q_names
            avg_quantile = zeros(T)
    
            for (i, forecaster) in enumerate(keys(forecasters_dfs))
                forecaster_q = forecasters_dfs[forecaster][:, q]
                avg_quantile = avg_quantile .+ weights[i] .* forecaster_q
            end
    
            avg_quantiles_df[:, q] = avg_quantile
        end
    
        return avg_quantiles_df
    
    end

    function calculate_quantile_loss_dataframe(y_true, y_preds, quantiles)

        if typeof(y_true) == Float64
            y_true = [y_true]
        end
    
        losses = Array{Float64}(undef, 1)
        for q in quantiles
            q_name = "q" * string(Int(q*100))
            loss = mean(QuantileLoss(q), y_preds[:, q_name], y_true)
            push!(losses, loss)
        end
    
        return mean(losses)
    
    end

    

    function get_avg_distribution(avg_quantiles::Array, dist::Distribution=Normal)

        avg_dist = fit(dist, avg_quantiles)
        return avg_dist

    end

end
