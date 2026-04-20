using LinearAlgebra
using DataStructures
using ProgressBars
using Base.Threads
using Plots.PlotMeasures
# using LossFunctions  # Removed as QuantileLoss is being replaced with custom quantile_loss
using Normalization
using RollingFunctions
using PlotlyJS
using Statistics
using Dates

include("../functions/functions.jl")
include("../functions/functions_payoff.jl")
include("../online_algorithms/quantile_regression.jl")
include("../online_algorithms/adaptive_robust_quantile_regression.jl")
include("../payoff/shapley_values.jl")
include("data_preprop.jl")
using .UtilsFunctions
using .UtilsFunctionsPayoff
using .QuantileRegression
using .Shapley
using .AdaptiveRobustRegression
using .RealWorldtestData

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

# Environment Settings
n_experiments = 100
T = 364
lead_time = 96
quantiles = [0.1, 0.5, 0.9]
total_reward = 100
delta = 0.7
algorithms = ["QR", "RQR"]
payoff_function = "Shapley"
burn_in_period = 60

root_dir = @__DIR__
plot_dir = joinpath(root_dir, "plots")
mkpath(plot_dir)
models_paths = OrderedDict(
    "ecmwf_nn" => joinpath(root_dir, "saved_models", "predictions_nn_ecmwf_ifs.parquet"),
    "noaa_nn" => joinpath(root_dir, "saved_models", "predictions_nn_noaa_gfs.parquet"),
    "dwd_nn" => joinpath(root_dir, "saved_models", "predictions_nn_dwd_icon_eu.parquet"),
    "ecmwf_xgb" => joinpath(root_dir, "saved_models", "predictions_xgb_ecmwf_ifs.parquet"),
    "noaa_xgb" => joinpath(root_dir, "saved_models", "predictions_xgb_noaa_gfs.parquet"),
    "dwd_xgb" => joinpath(root_dir, "saved_models", "predictions_xgb_dwd_icon_eu.parquet"),
    "ecmwf_qrf" => joinpath(root_dir, "saved_models", "predictions_qrf_ecmwf_ifs.parquet"),
    "noaa_qrf" => joinpath(root_dir, "saved_models", "predictions_qrf_noaa_gfs.parquet"),
    "dwd_qrf" => joinpath(root_dir, "saved_models", "predictions_qrf_dwd_icon_eu.parquet"),
)
model_names = collect(keys(models_paths))
n_forecasters = length(model_names)

# Environment Variables
realizations = OrderedDict([q => Vector{Vector{Float64}}() for q in quantiles])
algo_forecasts = OrderedDict([q => OrderedDict([name => [] for name in model_names]) for q in quantiles])
payoffs = OrderedDict([q => OrderedDict([algo => zeros(n_forecasters, T) for algo in algorithms]) for q in quantiles])
rewards = OrderedDict([q => OrderedDict([algo => zeros(n_forecasters, T) for algo in algorithms]) for q in quantiles])
weights = OrderedDict([q => OrderedDict([algo => zeros(n_forecasters, T) for algo in algorithms]) for q in quantiles])
rewards_in_sample = OrderedDict([q => OrderedDict([algo => zeros(n_forecasters, T) for algo in algorithms]) for q in quantiles])
rewards_out_sample = OrderedDict([q => OrderedDict([algo => zeros(n_forecasters, T) for algo in algorithms]) for q in quantiles])
losses_rqr = OrderedDict([q => zeros(T) for q in quantiles])
losses_qr = OrderedDict([q => zeros(T) for q in quantiles])

dates = Date[]

#################### Saving predictions and realizations ####################
data_lock = ReentrantLock()
for q in quantiles
    # Load data once per quantile, not once per time step
    true_prod, forecasters_preds, scalers, scaler_target, q_dates = preprocessing_forecasts(models_paths, q)
    if isempty(dates)
        global dates = q_dates
    end
    sorted_forecasters = copy(forecasters_preds)

    for t in ProgressBar(2:T)
        forecasters_preds_t = [sorted_forecasters[f][t] for f in model_names]
        y_true = true_prod[t]
        
        lock(data_lock) do
            if isempty(realizations[q])
                push!(realizations[q], zeros(length(y_true))) 
            end
            push!(realizations[q], copy(y_true))

            for name in model_names
                if isempty(algo_forecasts[q][name])
                    push!(algo_forecasts[q][name], zeros(length(y_true))) 
                end
                push!(algo_forecasts[q][name], denormalize(forecasters_preds[name][t], scalers[name])) 
            end
        end
    end
end

#################### Quantile Regression ####################
algo = "QR"
data_lock = ReentrantLock()

if algo in algorithms
    for q in quantiles
        quantile_step_reward = total_reward / length(quantiles)

        Threads.@threads for exp in ProgressBar(1:n_experiments)
            # Experiment Variables
            weights_exp = zeros(n_forecasters, T)
            payoffs_exp = zeros(n_forecasters, T)
            rewards_exp = zeros(n_forecasters, T)
            rewards_in_exp = zeros(n_forecasters, T)
            rewards_out_exp = zeros(n_forecasters, T)
            losses_qr_exp = zeros(T)

            weights_exp[:, 1] = initialize_weights(n_forecasters)

            # Data generation
            true_prod, forecasters_preds, scalers, scaler_target, _ = preprocessing_forecasts(models_paths, q)
            sorted_forecasters = copy(forecasters_preds)

            for t in 2:T
                forecasters_preds_t = [sorted_forecasters[f][t] for f in model_names]
                y_true = true_prod[t]

                y_true_sc = scaler_target(y_true)
                weights_exp[:, t], aggregated_forecast_t = online_quantile_regression_update_multiple_lead_times(forecasters_preds_t, weights_exp[:, t-1], y_true_sc, q, 0.1, 0.1)
                aggregated_forecast_t = denormalize(aggregated_forecast_t, scaler_target)
                # Loss calculation
                loss_t = mean(quantile_loss.(y_true, aggregated_forecast_t, q))
                losses_qr_exp[t] = loss_t
                
                denormalized_preds_t = [denormalize(forecasters_preds_t[i], scalers[model_names[i]]) for i in 1:n_forecasters]
                temp_payoffs = shapley_payoff_multiple_lead_times(denormalized_preds_t, weights_exp[:, t-1], y_true, q)
                forecasters_losses = [mean(quantile_loss.(y_true, denormalized_preds_t[i], q)) for i in 1:n_forecasters]
                temp_scores = 1 .- (forecasters_losses ./ sum(forecasters_losses))
                payoffs_exp[:, t] = payoff_update(payoffs_exp[:, t-1], temp_payoffs, 0.999)

                if t > burn_in_period
                    rewards_in = delta .* quantile_step_reward .* (max.(0, payoffs_exp[:, t]) ./ max(sum(max.(0, payoffs_exp[:, t])), eps()))
                    rewards_out = (1-delta) .* quantile_step_reward .* (temp_scores ./ sum(temp_scores))

                    rewards_in_exp[:, t] = rewards_in
                    rewards_out_exp[:, t] = rewards_out
                    rewards_exp[:, t] = rewards_in .+ rewards_out
                end
            end
            lock(data_lock) do
                payoffs[q][algo] += payoffs_exp
                weights[q][algo] += weights_exp
                rewards[q][algo] += rewards_exp
                rewards_in_sample[q][algo] += rewards_in_exp
                rewards_out_sample[q][algo] += rewards_out_exp
                losses_qr[q] += losses_qr_exp
            end
        end
    end

    for q in quantiles
        payoffs[q][algo] = payoffs[q][algo] ./ n_experiments
        weights[q][algo] = weights[q][algo] ./ n_experiments
        rewards[q][algo] = rewards[q][algo] ./ n_experiments
        rewards_in_sample[q][algo] = rewards_in_sample[q][algo] ./ n_experiments
        rewards_out_sample[q][algo] = rewards_out_sample[q][algo] ./ n_experiments
        losses_qr[q] = losses_qr[q] ./ n_experiments
    end
end

#################### Robust Quantile Regression ####################
algo = "RQR"
missing_rate = 0.05
data_lock = ReentrantLock()

if algo in algorithms
    for q in quantiles

        quantile_step_reward = total_reward / length(quantiles)

        Threads.@threads for exp in ProgressBar(1:n_experiments)
            # Experiment Variables
            weights_exp = zeros(n_forecasters, T)
            payoffs_exp = zeros(n_forecasters, T)
            rewards_exp = zeros(n_forecasters, T)
            rewards_in_exp = zeros(n_forecasters, T)
            rewards_out_exp = zeros(n_forecasters, T)
            losses_rqr_exp = zeros(T)

            weights_exp[:, 1] = initialize_weights(n_forecasters)

            # Data generation
            true_prod, forecasters_preds, scalers, scaler_target, _ = preprocessing_forecasts(models_paths, q)
            sorted_forecasters = copy(forecasters_preds)

            D_exp = zeros(n_forecasters, n_forecasters)
            alpha = Int.(rand(n_forecasters, T) .< missing_rate)
            for t in 1:T
                if sum(alpha[:, t]) == length(alpha[:, t])
                    idx = rand(1:length(alpha[:, t]))
                    alpha[idx, t] = 0
                end
            end

            for t in 2:T
                forecasters_preds_t = [sorted_forecasters[f][t] for f in model_names]
                y_true = true_prod[t]

                # Learning Phase
                y_true_sc = scaler_target(y_true)
                weights_exp[:, t], new_D, aggregated_forecast_t = online_adaptive_robust_quantile_regression_multiple_lead_times_trial(forecasters_preds_t, y_true_sc, weights_exp[:, t-1], D_exp, alpha[:, t], q, 0.1, 0.1)
                prev_D = D_exp
                D_exp = new_D
                aggregated_forecast_t = denormalize(aggregated_forecast_t, scaler_target)
                # Loss calculation
                loss_t = mean(quantile_loss.(y_true, aggregated_forecast_t, q))
                losses_rqr_exp[t] = loss_t

                # Payoff Calculation
                denormalized_preds_t = [denormalize(forecasters_preds_t[i], scalers[model_names[i]]) for i in 1:n_forecasters]
                temp_forecasts_t = [denormalized_preds_t[j] for j in 1:n_forecasters if alpha[j, t] == 0]
                temp_weights_t = weights_exp[:, t-1] .+ prev_D * alpha[:, t]
                temp_weights_t = [temp_weights_t[j] for j in 1:n_forecasters if alpha[j, t] == 0]
                temp_weights_t = project_to_simplex(temp_weights_t)

                temp_payoffs = nothing
                forecasters_losses = nothing
                if length(temp_forecasts_t) > 0
                    temp_payoffs = shapley_payoff_multiple_lead_times(temp_forecasts_t, temp_weights_t, y_true, q)
                    forecasters_losses = [mean(quantile_loss.(y_true, temp_forecasts_t[i], q)) for i in 1:length(temp_forecasts_t)]
                    temp_scores = [max(1 - (forecasters_losses[i] / sum(forecasters_losses)), eps()) for i in 1:length(forecasters_losses)]
                else 
                    temp_payoffs = zeros(n_forecasters)
                    forecasters_losses = ones(n_forecasters)
                end
                
                if length(temp_payoffs) < n_forecasters
                    for j in findall(a -> a == 1, alpha[:, t])
                        insert!(temp_payoffs, j, 0.0)
                        insert!(temp_scores, j, 0.0)
                    end
                end
                payoffs_exp[:, t] = payoff_update(payoffs_exp[:, t-1], temp_payoffs, 0.999)

                # Reward calculation
                if t > burn_in_period
                    rewards_in = zeros(n_forecasters)
                    rewards_out = zeros(n_forecasters)

                    payoffs_for_rewards_in = [max(0, payoffs_exp[j, t]) for j in 1:length(forecasters_preds_t) if alpha[j, t] == 0]
                    rewards_in[[j for j in 1:n_forecasters if alpha[j, t] == 1]] .= eps()
                    rewards_in[[j for j in 1:n_forecasters if alpha[j, t] == 0]] = delta .* quantile_step_reward .* (payoffs_for_rewards_in ./ max(sum(payoffs_for_rewards_in), eps()))

                    scores_for_rewards_out = [temp_scores[j] for j in 1:length(forecasters_preds_t) if alpha[j, t] == 0]
                    rewards_out[[j for j in 1:n_forecasters if alpha[j, t] == 1]] .= eps()
                    rewards_out[[j for j in 1:n_forecasters if alpha[j, t] == 0]] = (1-delta) .* quantile_step_reward .* (scores_for_rewards_out / sum(scores_for_rewards_out))

                    rewards_in_exp[:, t] = rewards_in
                    rewards_out_exp[:, t] = rewards_out
                    rewards_exp[:, t] = rewards_in .+ rewards_out
                end
            end

            lock(data_lock) do
                payoffs[q][algo] += payoffs_exp
                weights[q][algo] += weights_exp
                rewards[q][algo] += rewards_exp
                rewards_in_sample[q][algo] += rewards_in_exp
                rewards_out_sample[q][algo] += rewards_out_exp
                losses_rqr[q] += losses_rqr_exp
            end
        end
    end

    for q in quantiles
        payoffs[q][algo] = payoffs[q][algo] ./ n_experiments
        weights[q][algo] = weights[q][algo] ./ n_experiments
        rewards[q][algo] = rewards[q][algo] ./ n_experiments
        rewards_in_sample[q][algo] = rewards_in_sample[q][algo] ./ n_experiments
        rewards_out_sample[q][algo] = rewards_out_sample[q][algo] ./ n_experiments
        losses_rqr[q] = losses_rqr[q] ./ n_experiments
    end
end
    #################### Calculate Errors ####################
global_loss_qr = zeros(T-1)
global_loss_rqr = zeros(T-1)
global_loss_models = OrderedDict(name => zeros(T-1) for name in model_names)
individual_losses_by_quantile = OrderedDict(q => OrderedDict(name => Float64[] for name in model_names) for q in quantiles)

for q in quantiles

    global_loss_qr .+= losses_qr[q][2:T]
    global_loss_rqr .+= losses_rqr[q][2:T]

    individual_losses = OrderedDict(name => Float64[] for name in model_names)
    for t in 2:T
        y_true = realizations[q][t]

        for name in model_names
            y_pred = algo_forecasts[q][name][t]
            loss_t = mean(quantile_loss.(y_true, y_pred, q))

            global_loss_models[name][t-1] += loss_t
            push!(individual_losses[name], loss_t)
            push!(individual_losses_by_quantile[q][name], loss_t)
        end
    end
    println("\n############ RESULTS QUANTILE $q ############")
    println("Loss Aggregated QR : $(mean(losses_qr[q][(burn_in_period+1):T]))")
    println("Loss Aggregated RQR: $(mean(losses_rqr[q][(burn_in_period+1):T]))")
    
    for name in model_names
        avg_loss = mean(individual_losses[name][burn_in_period:end])
        println("Loss $(uppercase(name)): $avg_loss")
    end
end

########### Plots Settings ##############
model_colors_dict = Dict(
    "ecmwf_nn" => "green",
    "noaa_nn" => "orange",
    "dwd_nn" => "purple",
    "ecmwf_xgb" => "cyan",
    "noaa_xgb" => "magenta",
    "dwd_xgb" => "gold",
    "ecmwf_qrf" => "lime",
    "noaa_qrf" => "navy",
    "dwd_qrf" => "brown"
)

window = 7
model_name = "ecmwf_xgb"
cum_avg(v) = cumsum(v) ./ (1:length(v))
moving_avg(v, window) = runmean(v, window)

#################### PLOTTING AVERAGE LOSSES ####################
n_q = length(quantiles)
avg_ts_qr = global_loss_qr ./ n_q
avg_ts_rqr = global_loss_rqr ./ n_q
for name in model_names
    global_loss_models[name] ./= n_q
end

function make_loss_plot(avg_ts_qr, avg_ts_rqr, global_loss_models, model_name, window, dates)
    p = make_subplots(
        rows = 2,
        cols = 1,
        subplot_titles = reshape([
            "Moving Average Loss",
            "Instantaneous Loss",
        ], 2, 1),
        vertical_spacing = 0.08,
        shared_xaxes = true
    )

    # Helper to add all traces for a specific row and transformation
    function add_loss_traces!(p, transform_fn, row; show_legend=(row==1))
        # Preserve original line widths (thinner lines for row 3 instantaneous loss)
        w_agg = row == 2 ? 1.5 : 2.5
        w_mod = row == 2 ? 1.5 : 2.0

        # QR
        add_trace!(p, scatter(
            x = dates,
            y = transform_fn(avg_ts_qr),
            name = "QR",
            line = attr(color="blue", width=w_agg),
            legendgroup = "QR",
            showlegend = show_legend
        ), row=row, col=1)

        # RQR
        add_trace!(p, scatter(
            x = dates,
            y = transform_fn(avg_ts_rqr),
            name = "RQR",
            line = attr(color="red", width=w_agg),
            legendgroup = "RQR",
            showlegend = show_legend
        ), row=row, col=1)

        # Single Model 
        if haskey(global_loss_models, model_name)
            add_trace!(p, scatter(
                x = dates,
                y = transform_fn(global_loss_models[model_name]),
                name = "$(uppercase(model_name))",
                line = attr(color="black", width=w_mod),
                opacity = 0.8,
                legendgroup = model_name,
                showlegend = show_legend
            ), row=row, col=1)
        elseif row == 1 # Print warning only once to avoid spamming the console
            println("Warning: Model $model_name not found in global_loss_models")
        end
    end

    # Add Moving Average (Row 1)
    add_loss_traces!(p, v -> moving_avg(v, window), 1)
    # Add Instantaneous (Row 2)
    add_loss_traces!(p, identity, 2)

    # Layout matched exactly to the reward plot
    relayout!(p,
        height = 1200,
        width  = 1800,
        xaxis2_tickformat = "%b",
        xaxis2_tickfont = attr(size=14),
        xaxis2_title = attr(text="Time Step", font=attr(size=18)),
        hovermode = "x unified",
        legend = attr(
            orientation = "h",
            yanchor = "top",
            y = -0.13,
            xanchor = "center",
            x = 0.5,
            font = attr(size = 16)
        ),
        margin = attr(l=55, r=10, t=40, b=100)
    )

    # Adding shared ylabel annotation
    existing_annotations = p.plot.layout[:annotations]
    ylabel_annotation = attr(
        text = "Avg Quantile Loss",
        x = -0.07,
        xref = "paper",
        y = 0.5,
        yref = "paper",
        showarrow = false,
        textangle = -90,
        font_size = 18,
        xanchor = "center",
        yanchor = "middle"
    )
    relayout!(p, annotations = vcat(existing_annotations, [ylabel_annotation]))

    return p
end

# Generate, save, and display
p_loss = make_loss_plot(avg_ts_qr[burn_in_period:end], avg_ts_rqr[burn_in_period:end], 
                        OrderedDict(k => v[burn_in_period:end] for (k,v) in global_loss_models), 
                        model_name, window, dates[(burn_in_period+1):T])

savefig(p_loss, joinpath(plot_dir, "loss_analysis.pdf"))
display(p_loss)

#################### PLOTTING WEIGHTS BY QUANTILE ####################
n_rows = length(quantiles)

# 1. Initialize Subplots
p_weights_combined = make_subplots(
    rows = n_rows, 
    cols = 1,
    subplot_titles = reshape(["Weights for Quantile $q" for q in quantiles], n_rows, 1),
    vertical_spacing = 0.10, 
    shared_xaxes = true
)

# 2. Iterate through quantiles and models
for (r, q) in enumerate(quantiles)
    for (i, name) in enumerate(model_names)
        show_leg = (r == 1)
        color = get(model_colors_dict, name, "gray")
        trace = scatter(
            #y = weights[q]["RQR"][i, :],
            x = dates[(burn_in_period+1):T],
            y = moving_avg(weights[q]["RQR"][i, (burn_in_period+1):T], window),
            name = uppercase(name),
            mode = "lines",
            line = attr(width=1.5, color=color), # Slightly thicker for visibility
            legendgroup = name,
            showlegend = show_leg
        )
        
        add_trace!(p_weights_combined, trace, row=r, col=1)
    end
end

# 3. Configure the Global Layout and Legend
total_height = 500 * n_rows 
layout_updates = Dict{Symbol, Any}(
    :height => total_height, 
    :width => 1200,
    
    # LEGEND CONFIGURATION
    :legend => attr(
        orientation = "h",      
        yanchor = "top",       
        y = -0.2,  # Pushes legend down below the x-axis label
        xanchor = "center",     
        x = 0.5,
        bgcolor = "rgba(0,0,0,0)", # Transparent background
        font = attr(size=14)
    ),
    
    # MARGINS
    :margin => attr(l=55, r=10, t=40, b=110)
)

# Set X-axis label on the very last subplot
last_xaxis_name = (n_rows == 1) ? "xaxis" : "xaxis$(n_rows)"
layout_updates[Symbol("$(last_xaxis_name)_tickformat")] = "%b"
layout_updates[Symbol("$(last_xaxis_name)_title")] = attr(text="Time Step", font=attr(size=18))
layout_updates[Symbol("$(last_xaxis_name)_title_standoff")] = 20
layout_updates[Symbol("$(last_xaxis_name)_tickfont")] = attr(size=14)

relayout!(p_weights_combined, layout_updates)

# Add shared y-label annotation separately — preserve existing subplot annotations
existing_annotations = p_weights_combined.plot.layout[:annotations]
ylabel_annotation = attr(
    text = "Weights",
    x = -0.07,
    xref = "paper",
    y = 0.5,
    yref = "paper",
    showarrow = false,
    textangle = -90,
    font_size = 18,
    xanchor = "center",
    yanchor = "middle"
)

relayout!(p_weights_combined, 
    annotations = vcat(existing_annotations, [ylabel_annotation])
)

# 4. Save and Display
savefig(p_weights_combined, joinpath(plot_dir, "weights_distribution.pdf"))
display(p_weights_combined)

#################### Plot Results ####################
total_rewards_forecasters = OrderedDict([algo => zeros(n_forecasters, T) for algo in algorithms])
total_in_rewards_forecasters = OrderedDict([algo => zeros(n_forecasters, T) for algo in algorithms])
total_out_rewards_forecasters = OrderedDict([algo => zeros(n_forecasters, T) for algo in algorithms])
for algo in algorithms
    for i in 1:n_forecasters
        for q in quantiles
            total_rewards_forecasters[algo][i, :] .+= rewards[q][algo][i, :]
            total_in_rewards_forecasters[algo][i, :] .+= rewards_in_sample[q][algo][i, :]
            total_out_rewards_forecasters[algo][i, :] .+= rewards_out_sample[q][algo][i, :]
        end
    end
end

#################### PLOTTING TOTAL REWARDS ####################

# Per-model time series for QR and RQR
model_rewards_qr  = OrderedDict(model_names[i] => total_rewards_forecasters["QR"][i,  (burn_in_period+1):T] for i in 1:n_forecasters)
model_rewards_rqr = OrderedDict(model_names[i] => total_rewards_forecasters["RQR"][i, (burn_in_period+1):T] for i in 1:n_forecasters)

function make_reward_plot(model_rewards, window, dates)
    p = make_subplots(
        rows = 2,
        cols = 1,
        subplot_titles = reshape([
            "Moving Average Reward",
            "Instantaneous Reward",
        ], 2, 1),
        vertical_spacing = 0.08,
        shared_xaxes = true
    )

    # Helper to add all traces to a given row
    function add_all_traces!(p, transform_fn, row; show_legend=(row==1))

        # Per-model
        for name in model_names
            color = get(model_colors_dict, name, "gray")
            add_trace!(p, scatter(
                x = dates,
                y = transform_fn(model_rewards[name]),
                name = uppercase(name),
                line = attr(color=color, width=1.5),
                opacity = 0.8,
                legendgroup = name,
                showlegend = show_legend
            ), row=row, col=1)
        end
    end

    add_all_traces!(p, v -> moving_avg(v, window), 1)
    add_all_traces!(p, identity, 2)

    relayout!(p,
        height = 1200,
        width  = 1800,
        xaxis2_tickformat = "%b",
        xaxis2_tickfont = attr(size=14),
        xaxis2_title = attr(text="Time Step", font=attr(size=18)),
        hovermode = "x unified",
        legend = attr(
            orientation = "h",
            yanchor = "top",
            y = -0.13,
            xanchor = "center",
            x = 0.5,
            font = attr(size = 14)
        ),
        margin = attr(l=55, r=10, t=40, b=100)
    )

    # Adding shared ylabel
    existing_annotations = p.plot.layout[:annotations]
    ylabel_annotation = attr(
        text = "Rewards (£)",
        x = -0.07,
        xref = "paper",
        y = 0.5,
        yref = "paper",
        showarrow = false,
        textangle = -90,
        font_size = 18,
        xanchor = "center",
        yanchor = "middle"
    )

    relayout!(p, annotations = vcat(existing_annotations, [ylabel_annotation]))

    return p
end

p_reward_qr  = make_reward_plot(model_rewards_qr,  window, dates[(burn_in_period+1):T])
p_reward_rqr = make_reward_plot(model_rewards_rqr, window, dates[(burn_in_period+1):T])

savefig(p_reward_qr,  joinpath(plot_dir, "reward_analysis_qr.pdf"))
savefig(p_reward_rqr, joinpath(plot_dir, "reward_analysis_rqr.pdf"))
display(p_reward_qr)
display(p_reward_rqr)

#################### Print Monthly Rewards ####################
println("\n############ MONTHLY REWARDS (Excluding Burn-in) ############")

for algo in algorithms
    println("\nAlgorithm: $algo")
    
    # Aggregate rewards by month
    monthly_rewards = OrderedDict{Tuple{Int, Int}, Vector{Float64}}()
    
    for t in (burn_in_period + 1):T
        d = dates[t]
        ym = (year(d), month(d))
        
        if !haskey(monthly_rewards, ym)
            monthly_rewards[ym] = zeros(n_forecasters)
        end
        
        monthly_rewards[ym] .+= total_rewards_forecasters[algo][:, t]
    end
    
    # Print results
    for (ym, rewards_vec) in monthly_rewards
        m_name = monthname(ym[2])
        y_val = ym[1]
        println("  Period: $m_name $y_val")
        for (i, name) in enumerate(model_names)
            println("    $(uppercase(name)): $(round(rewards_vec[i], digits=2))")
        end
    end
end