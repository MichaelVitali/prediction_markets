module DataGeneration
using Statistics
using Distributions

export generate_time_invariant_data, generate_time_variant_data

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

end