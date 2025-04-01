using Revise
using Plots
using StatsPlots
using Distributions

include("data_generation/DataGeneration.jl")
include("functions/functions.jl")
using .DataGeneration
using .UtilsFunctions

data_inv, _ = generate_time_invariant_data(3, [1.0, 2.0, 3.0])
data_var = generate_time_variant_data(1000)

println("Mean of Y: ", mean(data_inv))
println("Variance of Y: ", var(data_inv))

#=plot(
    plot(data_inv, title="Time Invariant Data", linewidth=2, legend=false),
    plot(data_var, title="Time Variant Data", linewidth=2, legend=false),
    layout=(2, 1),  # 1 row, 2 columns (side by side)
    size=(400, 600)  # Adjust overall figure size
)=#

#=forecasters_dists = [
    Dict(
        "f1" => Normal(20, 5),
        "f2" => Normal(40, 5),
        "f3" => Normal(60, 10),
    ),
    Dict(
        "f1" => Normal(20, 5),
        "f2" => Normal(40, 5),
        "f3" => Normal(60, 10),
    )
]
quantiles = [0.1, 0.5, 0.9]
weights = [1, 1, 1]

trial = quantile_averaging_dist_multiple_times(forecasters_dists, quantiles, weights, 2)
display(trial)=#