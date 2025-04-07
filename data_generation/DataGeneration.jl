module DataGeneration
using Statistics
using Distributions

export generate_time_invariant_data, generate_time_variant_data, data_generation, create_forecasters_preds

    function generate_time_invariant_data(n_samples::Int, coefs::Vector{Float64})
        n_forecasters = length(coefs)
        
        # Generate base variables
        X0 = randn(n_samples)
        X_private = [randn(n_samples) for _ in 1:n_forecasters]
        eps = randn(n_samples)
        
        Y = X0 .+ eps
        for (i, c) in enumerate(coefs)
            Y .+= (c .* X_private[i])
        end
        
        # Package distributions in Dict
        distributions = Dict("X0" => X0)
        for i in 1:n_forecasters
            distributions["X$i"] = X_private[i]
        end
        
        return Y, distributions
    end

    function generate_time_variant_data(
        T::Integer=1000,
        mu0::Float64=1.0,
        std::Float64=1.0,
    )

        Y_T = Array{Float64}(undef, T)
        eps = randn(T)

        # t = 0
        mean_0 = 0.15 * asin(mu0)
        Y_T[1] = rand(Normal(mean_0, std))

        mu_t = mu0
        for t in range(2, T)
            mu_t = 0.99 * mu_t + eps[t]
            mean_t = 0.15 * asinh(mu_t)
            Y_T[t] = rand(Normal(mean_t, std))
        end

        return Y_T

    end

    function data_generation(case_study, T, q)

        if case_study == "Gneiting"
            X0 = randn(T)
            X1 = randn(T)
            X2 = randn(T)
            X3 = randn(T)
            e = randn(T)
    
            a1 = 1
            a2 = 1
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
                Y[t] = quantile(Normal((0.15*asinh(mu)),1), q)
            end
    
            forecasters = Dict([
                "f1" => [quantile(Normal(-1, 1), q) for t in 1:T],
                "f2" => [quantile(Normal(3, 2), q) for t in 1:T]
            ])
        end
    
        return Y, forecasters
    end

    function create_forecasters_preds(
        distributions,
        q,
        coefs::Vector{Float64},
    )
    
        forecasters_dict = Dict()
    
        for (i, coef) in enumerate(coefs)
            forecaster_name = "f" * string(i)
    
            other_coefs = vcat(coefs[1:i-1], coefs[i+1:end])
            forecaster_mean = distributions["X0"] .+ (coef .* distributions["X$i"])
            forecaster_var = 1 + sum(other_coefs.^2)
    
            forecaster_dists = Normal.(forecaster_mean, sqrt(forecaster_var))
            forecasters_dict[forecaster_name] = quantile.(forecaster_dists, q)
        end
    
        return forecasters_dict
    
    end

end