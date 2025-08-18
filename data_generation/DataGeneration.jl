module DataGeneration
using Statistics
using Distributions
using LinearAlgebra

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

    function generate_time_invariant_data_multiple_lead_times(T, n, q)
    
    forecasters_dict = Dict("f1" => [], "f2" => [], "f3" => [])
    true_values = []
    true_weights = zeros(3, T)

    for i in 1:T
        mu1 = zeros(n).+ randn(n).*0.5
        mu2 = fill(1, n).+ randn(n).*0.5
        mu3 = fill(2, n).+ randn(n).*0.5
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

        push!(forecasters_dict["f1"], f1)
        push!(forecasters_dict["f2"], f2)
        push!(forecasters_dict["f3"], f3)
        push!(true_values, rand.(Y))
        true_weights[:, i] = w
    end

    return true_values, forecasters_dict, true_weights
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

    function generate_abrupt_data_multiple_lead_times(T, n, q)

        forecasters_dict = Dict("f1" => [], "f2" => [], "f3" => [])
        true_values = []
        true_weights = zeros(3, T)
        w1 = [0.1, 0.6, 0.3]
        w2 = [0.4, 0.2, 0.4]

        for i in 1:T
            mu1 = zeros(n).+ randn(n).*0.5
            mu2 = fill(1, n).+ randn(n).*0.5
            mu3 = fill(2, n).+ randn(n).*0.5
            sig1 = 1
            sig2 = 1
            sig3 = 1

            if i < T/2
                w = w1
            else
                w = w2
            end

            F1 = Normal.(mu1, sig1)
            F2 = Normal.(mu2, sig2)
            F3 = Normal.(mu3, sig3)

            mu_y = mu1 .* w[1] .+ mu2 .* w[2] .+ mu3 .* w[3]
            sig_y = w[1] * sig1 + w[2] * sig2 + w[3] * sig3
            Y = Normal.(mu_y, sig_y)

            f1 = quantile.(F1, q)
            f2 = quantile.(F2, q)
            f3 = quantile.(F3, q)

            push!(forecasters_dict["f1"], f1)
            push!(forecasters_dict["f2"], f2)
            push!(forecasters_dict["f3"], f3)
            push!(true_values, rand.(Y))
            true_weights[:, i] = w
        end

        return true_values, forecasters_dict, true_weights

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

    function generate_dynamic_data_sin_multiple_lead_times(T, n, q)

        forecasters_dict = Dict("f1" => [], "f2" => [], "f3" => [])
        true_values = []
        true_weights = zeros(3, T)
        w1 = [0.1, 0.6, 0.3]
        w2 = [0.4, 0.2, 0.4]
        lambda = 0.999
        w = [1/3, 1/3, 1/3]
        cycles = 1

        for i in 1:T
            mu1 = zeros(n).+ randn(n).*0.5
            mu2 = fill(1, n).+ randn(n).*0.5
            mu3 = fill(2, n).+ randn(n).*0.5
            sig1 = 1
            sig2 = 1
            sig3 = 1

            alpha = 0.5 * (1 .+ sin(2 * pi * cycles * i / T))  # n full cycles over time T
            w_target = (1 .- alpha) .* w1 .+ alpha .* w2
            w = lambda .* w .+ (1 - lambda) .* w_target

            F1 = Normal.(mu1, sig1)
            F2 = Normal.(mu2, sig2)
            F3 = Normal.(mu3, sig3)

            mu_y = mu1 .* w[1] .+ mu2 .* w[2] .+ mu3 .* w[3]
            sig_y = w[1] * sig1 + w[2] * sig2 + w[3] * sig3
            Y = Normal.(mu_y, sig_y)

            f1 = quantile.(F1, q)
            f2 = quantile.(F2, q)
            f3 = quantile.(F3, q)

            push!(forecasters_dict["f1"], f1)
            push!(forecasters_dict["f2"], f2)
            push!(forecasters_dict["f3"], f3)
            push!(true_values, rand.(Y))
            true_weights[:, i] = w
        end

        return true_values, forecasters_dict, true_weights

    end

end