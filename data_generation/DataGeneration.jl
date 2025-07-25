module DataGeneration
using Statistics
using Distributions
using LinearAlgebra

export data_generation_case_study_abrupt, data_generation_case_study_invariant, generate_time_invariant_data, data_generation_case_study_dynamic, generate_abrupt_data, generate_dynamic_data, generate_dynamic_data_sin, generate_time_invariant_data_multiple_lead_times

    function data_generation_case_study_invariant(case_study, T, q)

        if case_study == "Gneiting"
            X0 = randn(T)
            X1 = randn(T)
            X2 = randn(T)
            X3 = randn(T)
            e = randn(T)
    
            a1 = 1.0
            a2 = 1.0
            a3 = 1.1
    
            Y = X0 .+ (a1 .* X1) .+ (a2 .* X2) .+ (a3 .* X3) .+ e
    
            MU1 = X0 .+ a1.*X1
            MU2 = X0 .+ a2.*X2
            MU3 = X0 .+ a3.*X3
            sd1 = (1+a2^2+a3^2)^0.5
            sd2 = (1+a1^2+a3^2)^0.5
            sd3 = (1+a1^2+a2^2)^0.5
    
            forecasters = Dict([
                "f1" => quantile.(Normal.(MU1, sd1), q),
                "f2" => quantile.(Normal.(MU2, sd2), q),
                "f3" => quantile.(Normal.(MU3, sd3), q),
            ])

        elseif case_study == "Berrisch"
            mu = 0
            Y = zeros(T)
    
            for t in 1:T
                mu = 0.99 * mu + randn(1)[1]
                Y[t] = rand(Normal((0.15*asinh(mu)),1))
            end
    
            forecasters = Dict([
                "f1" => [quantile(Normal(-1, 1), q) for t in 1:T],
                "f2" => [quantile(Normal(3, 2), q) for t in 1:T]
            ])

        end
    
        return Y, forecasters
    end

    function data_generation_case_study_abrupt(case_study, T, q)

        if case_study == "Gneiting"
            X0 = randn(T)
            X1 = randn(T)
            X2 = randn(T)
            X3 = randn(T)
            e = randn(T)
            Y = zeros(T)
            f1 = zeros(T)
            f2 = zeros(T)
            f3 = zeros(T)

            for t in 1:T

                if t < (T/2)
                    a1 = 1
                    a2 = 1
                    a3 = 1.1
                else
                    a1 = 1.5
                    a2 = 1.1
                    a3 = 1
                end

                Y[t] = X0[t] + (a1 * X1[t]) + (a2 * X2[t]) + (a3 * X3[t]) + e[t]

                MU1 = X0[t] .+ a1.*X1[t]
                MU2 = X0[t] .+ a2.*X2[t]
                MU3 = X0[t] .+ a3.*X3[t]
                sd1 = (1+a2^2+a3^2)^0.5
                sd2 = (1+a1^2+a3^2)^0.5
                sd3 = (1+a1^2+a2^2)^0.5

                f1[t] = quantile(Normal(MU1, sd1), q)
                f2[t] = quantile(Normal(MU2, sd2), q)
                f3[t] = quantile(Normal(MU3, sd3), q)
            end
    
            forecasters = Dict([
                "f1" => f1,
                "f2" => f2,
                "f3" => f3,
            ])
        end

        return Y, forecasters
    end

    function data_generation_case_study_dynamic(case_study, T, q)

        if case_study == "Gneiting"
            X0 = randn(T)
            X1 = randn(T)
            X2 = randn(T)
            X3 = randn(T)
            e = randn(T)
            Y = zeros(T)
            f1 = zeros(T)
            f2 = zeros(T)
            f3 = zeros(T)

            a1 = 1
            a2 = 1
            a3 = 1.1
            lambda = 0.9999

            for t in 1:T

                Y[t] = X0[t] + (a1 * X1[t]) + (a2 * X2[t]) + (a3 * X3[t]) + e[t]

                MU1 = X0[t] .+ a1.*X1[t]
                MU2 = X0[t] .+ a2.*X2[t]
                MU3 = X0[t] .+ a3.*X3[t]
                sd1 = (1+a2^2+a3^2)^0.5
                sd2 = (1+a1^2+a3^2)^0.5
                sd3 = (1+a1^2+a2^2)^0.5

                f1[t] = quantile(Normal(MU1, sd1), q)
                f2[t] = quantile(Normal(MU2, sd2), q)
                f3[t] = quantile(Normal(MU3, sd3), q)

                if t < T/4 || (t >= T/2 && t < 3/4*T)
                    a1 = lambda * a1 + (1-lambda) * 1.5
                    a2 = lambda * a2 + (1-lambda) * 1.1
                    a3 = lambda * a3 + (1-lambda) * 1.2
                elseif t >= T/4 && t < T/2 || (t >= 3/4*T)
                    a1 = lambda * a1 + (1-lambda) * 1
                    a2 = lambda * a2 + (1-lambda) * 1
                    a3 = lambda * a3 + (1-lambda) * 1.1
                end
            end
    
            forecasters = Dict([
                "f1" => f1,
                "f2" => f2,
                "f3" => f3,
            ])
        end

        return Y, forecasters
    end

    function generate_time_invariant_data(T, q)
        
        mu1 = zeros(T).+ randn(T).*0.5
        mu2 = fill(1, T).+ randn(T).*0.5
        mu3 = fill(2, T).+ randn(T).*0.5
        sig1 = 1
        sig2 = 1
        sig3 = 1
        w = [0.1, 0.6, 0.3]
        #w = [0.4, 0.2, 0.4]

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

    function generate_dynamic_data_sin(T, q, n=4)

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

            alpha = 0.5 * (1 .+ sin(2 * pi * n * t / T))  # n full cycles over time T

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

end