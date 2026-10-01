# Environment settings shared by the real-world test drivers (main_rewards.jl, main_missing_rates.jl).
# Script-specific settings (number of Monte-Carlo runs, reward or missing-rate parameters) stay in each driver.
using DataStructures

T = 364
quantiles = [0.1, 0.5, 0.9]
burn_in_period = 60
lr = 0.1
batch_percentage = 0.1
lower_bound_mw = 0.0
upper_bound_mw = 2262.1
seed = 1234
window = 7                  # moving-average window [sessions] for the plots

root_dir = @__DIR__
plot_dir = joinpath(root_dir, "plots")
results_dir = joinpath(root_dir, "results")
mkpath(plot_dir)
mkpath(results_dir)

models_paths = OrderedDict(
    "ecmwf_qrf" => joinpath(root_dir, "saved_models", "predictions_qrf_ecmwf_ifs.parquet"),
    "ecmwf_mlp" => joinpath(root_dir, "saved_models", "predictions_nn_ecmwf_ifs.parquet"),
    "ecmwf_xgb" => joinpath(root_dir, "saved_models", "predictions_xgb_ecmwf_ifs.parquet"),
    "noaa_qrf" => joinpath(root_dir, "saved_models", "predictions_qrf_noaa_gfs.parquet"),
    "noaa_mlp" => joinpath(root_dir, "saved_models", "predictions_nn_noaa_gfs.parquet"),
    "noaa_xgb" => joinpath(root_dir, "saved_models", "predictions_xgb_noaa_gfs.parquet"),
    "dwd_qrf" => joinpath(root_dir, "saved_models", "predictions_qrf_dwd_icon_eu.parquet"),
    "dwd_mlp" => joinpath(root_dir, "saved_models", "predictions_nn_dwd_icon_eu.parquet"),
    "dwd_xgb" => joinpath(root_dir, "saved_models", "predictions_xgb_dwd_icon_eu.parquet"),
)
model_names = collect(keys(models_paths))
n_forecasters = length(model_names)
