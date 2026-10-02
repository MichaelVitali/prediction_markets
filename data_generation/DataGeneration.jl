module DataGeneration
using Statistics
using Distributions
using LinearAlgebra
using Random

export generate_time_invariant_data, generate_abrupt_data, generate_dynamic_data, generate_dynamic_data_sin, generate_time_invariant_data_multiple_lead_times, generate_abrupt_data_multiple_lead_times, generate_dynamic_data_sin_multiple_lead_times

    

    function generate_time_invariant_data(T, q)
        
        mu1 = zeros(T).+ randn(T).*0.5
        mu2 = fill(1, T).+ randn(T).*0.5
        mu3 = fill(2, T).+ randn(T).*0.5
        sig1 = 1
        sig2 = 1
        sig3 = 1
        w = [0.1, 0.6, 0.3]

        F1 = Normal.(mu1, sig1)
        F2 = Normal.(mu2, sig2)
        F3 = Normal.(mu3, sig3)

        mu_y = mu1 .* w[1] .+ mu2 .* w[2] .+ mu3 .* w[3]
        sig_y = w[1] * sig1 + w[2] * sig2 + w[3] * sig3
        Y = Normal.(mu_y, sig_y)

        f1 = quantile.(F1, q)
        f2 = quantile.(F2, q)
        f3 = quantile.(F3, q)

        forecasters_dict = Dict("f1" => f1, "f2" => f2, "f3" => f3)
        true_values = rand.(Y)

        true_weights = repeat(w', T)
            
        return true_values, forecasters_dict, true_weights
    end

    """
        simulate_sessions(T, n, quantiles, rng, weights_at)

    Shared core of the `*_multiple_lead_times` generators. In every session, each forecaster's predictive
    distribution N(mu_k, 1) is drawn once (one mean per lead time, centred at 0, 1, 2) and *all* quantile
    forecasts are taken from that same distribution, so they never cross. The realization is drawn once
    per session from N(sum_k w_k mu_k, sum_k w_k), with `w = weights_at(i)` the true weights at session i.

    Returns `(true_values, forecasts, true_weights)`: `true_values[i]` (one value per lead time),
    `forecasts[q]["f1".."f3"][i]` and the `(3, T)` matrix of true weights.
    """
    function simulate_sessions(T, n, quantiles, rng, weights_at)
        centers = [0, 1, 2]
        sigmas = [1, 1, 1]
        forecasts = Dict(q => Dict("f$k" => Vector{Vector{Float64}}() for k in 1:3) for q in quantiles)
        true_values = Vector{Vector{Float64}}()
        true_weights = zeros(3, T)

        for i in 1:T
            mus = [centers[k] .+ randn(rng, n) .* 0.5 for k in 1:3]   # one draw per session, shared by all quantiles
            w = weights_at(i)

            for q in quantiles, k in 1:3
                push!(forecasts[q]["f$k"], quantile.(Normal.(mus[k], sigmas[k]), q))
            end

            mu_y = mus[1] .* w[1] .+ mus[2] .* w[2] .+ mus[3] .* w[3]
            sig_y = w[1] * sigmas[1] + w[2] * sigmas[2] + w[3] * sigmas[3]
            push!(true_values, rand.(Ref(rng), Normal.(mu_y, sig_y)))
            true_weights[:, i] = w
        end

        return true_values, forecasts, true_weights
    end

    # Constant true weights
    function generate_time_invariant_data_multiple_lead_times(T, n, quantiles; rng=Random.default_rng())
        return simulate_sessions(T, n, quantiles, rng, i -> [0.1, 0.6, 0.3])
    end

    function generate_abrupt_data(T, q)

        mu1 = zeros(T) .+ randn(T).*0.5
        mu2 = fill(1, T) .+ randn(T).*0.5
        mu3 = fill(2, T) .+ randn(T).*0.5
        sig1 = 1
        sig2 = 1
        sig3 = 1

        F1 = Normal.(mu1, sig1)
        F2 = Normal.(mu2, sig2)
        F3 = Normal.(mu3, sig3)

        true_values = zeros(T)
        w1 = [0.1, 0.6, 0.3]
        w2 = [0.4, 0.2, 0.4]

        for t in 1:T
            if t < T/2
                w = w1
            else
                w = w2
            end
            
            mu_y = mu1[t] .* w[1] .+ mu2[t] .* w[2] .+ mu3[t] .* w[3]
            sig_y = w[1] * sig1 + w[2] * sig2 + w[3] * sig3
            Y = Normal(mu_y, sig_y)
            true_values[t] = rand(Y)
        end

        f1 = quantile.(F1, q)
        f2 = quantile.(F2, q)
        f3 = quantile.(F3, q)

        forecasters_dict = Dict("f1" => f1, "f2" => f2, "f3" => f3)

        true_weights = Matrix{Float16}(undef, T, 3)
        true_weights[1:map(Int, T/2), :] .= w1'
        true_weights[map(Int, T/2)+1:end, :] .= w2'
            
        return true_values, forecasters_dict, true_weights
    end

    # True weights jump from w1 to w2 halfway through
    function generate_abrupt_data_multiple_lead_times(T, n, quantiles; rng=Random.default_rng())
        w1 = [0.1, 0.6, 0.3]
        w2 = [0.4, 0.2, 0.4]
        return simulate_sessions(T, n, quantiles, rng, i -> i < T/2 ? w1 : w2)
    end

    function generate_dynamic_data(T, q, n=4)

        mu1 = zeros(T) .+ randn(T)*0.5
        mu2 = fill(1, T) .+ randn(T)*0.5
        mu3 = fill(2, T) .+ randn(T)*0.5
        sig1 = 1
        sig2 = 1
        sig3 = 1

        F1 = Normal.(mu1, sig1)
        F2 = Normal.(mu2, sig2)
        F3 = Normal.(mu3, sig3)

        true_values = zeros(T)
        true_weights = Matrix{Float16}(undef, T, 3)
        w1 = [0.1, 0.6, 0.3]
        w2 = [0.4, 0.2, 0.4]
        lambda = 0.999
        w = [0.4, 0.2, 0.4]

        for t in 1:T

            split_index = floor(Int, t / T * n)

            if split_index % 2 == 0
                w  = lambda .* w + (1-lambda) .* w1
            else
                w  = lambda .* w + (1-lambda) .* w2
            end
            
            mu_y = mu1[t] .* w[1] .+ mu2[t] .* w[2] .+ mu3[t] .* w[3]
            sig_y = w[1] * sig1 + w[2] * sig2 + w[3] * sig3
            Y = Normal(mu_y, sig_y)
            true_values[t] = rand(Y)

            true_weights[t, :] = w
        end

        f1 = quantile.(F1, q)
        f2 = quantile.(F2, q)
        f3 = quantile.(F3, q)

        forecasters_dict = Dict("f1" => f1, "f2" => f2, "f3" => f3)
            
        return true_values, forecasters_dict, true_weights
    end

    function generate_dynamic_data_sin(T, q, cycles=4)

        mu1 = zeros(T) .+ randn(T)*0.5
        mu2 = fill(1, T) .+ randn(T)*0.5
        mu3 = fill(2, T) .+ randn(T)*0.5
        sig1 = 1
        sig2 = 1
        sig3 = 1

        F1 = Normal.(mu1, sig1)
        F2 = Normal.(mu2, sig2)
        F3 = Normal.(mu3, sig3)

        true_values = zeros(T)
        true_weights = Matrix{Float16}(undef, T, 3)
        w1 = [0.1, 0.6, 0.3]
        w2 = [0.4, 0.2, 0.4]
        lambda = 0.999
        w = [1/3, 1/3, 1/3]

        for t in 1:T

            alpha = 0.5 * (1 .+ sin(2 * pi * cycles * t / T))  # n full cycles over time T

            # Interpolate between w1 and w2 using alpha
            w_target = (1 .- alpha) .* w1 .+ alpha .* w2

            # Apply exponential smoothing to approach the target smoothly
            w = lambda .* w .+ (1 - lambda) .* w_target

            true_weights[t, :] = w
            
            mu_y = mu1[t] .* w[1] .+ mu2[t] .* w[2] .+ mu3[t] .* w[3]
            sig_y = w[1] * sig1 + w[2] * sig2 + w[3] * sig3
            Y = Normal(mu_y, sig_y)
            true_values[t] = rand(Y)

            true_weights[t, :] = w
        end

        f1 = quantile.(F1, q)
        f2 = quantile.(F2, q)
        f3 = quantile.(F3, q)

        forecasters_dict = Dict("f1" => f1, "f2" => f2, "f3" => f3)
            
        return true_values, forecasters_dict, true_weights
    end

    # True weights follow a smoothed (EMA) sinusoidal path between w1 and w2
    function generate_dynamic_data_sin_multiple_lead_times(T, n, quantiles; rng=Random.default_rng())
        w1 = [0.1, 0.6, 0.3]
        w2 = [0.4, 0.2, 0.4]
        lambda = 0.999
        cycles = 1
        w = [1/3, 1/3, 1/3]
        function weights_at(i)
            alpha = 0.5 * (1 .+ sin(2 * pi * cycles * i / T))  # n full cycles over time T
            w_target = (1 .- alpha) .* w1 .+ alpha .* w2
            w = lambda .* w .+ (1 - lambda) .* w_target
            return w
        end
        return simulate_sessions(T, n, quantiles, rng, weights_at)
    end

end