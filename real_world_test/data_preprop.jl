module RealWorldtestData

using DataFrames
using Parquet2
using Dates
using Normalization
using CSV
using DataStructures

export preprocessing_forecasts

    function preprocessing_forecasts(model_paths, q, lower_bound, upper_bound)
        lower_bound < upper_bound || throw(ArgumentError("lower_bound must be smaller than upper_bound"))

        forecasters_dict = OrderedDict{String, Vector{Vector{Float64}}}()
        true_values = []
        dates = Date[]
        # A forecast and its realization must have the same physical meaning on
        # the normalized scale, regardless of which model produced it.
        scaler = MinMax([Float64(lower_bound), Float64(upper_bound)])

        for name in keys(model_paths)
            forecasters_dict[name] = []
        end

        ######## Extract MODELS FORECASTS ########
        col_sym = Symbol("q$(Int(q * 100))")   
        for (name, path) in model_paths
            df = DataFrame(Parquet2.Dataset(path))
            df = dropmissing(df)
            df[!, col_sym] = clamp.(df[!, col_sym], lower_bound, upper_bound)

            first_date = Date(first(df.datetime))
            last_date = Date(last(df.datetime)) - Day(1)
            for day in first_date:Day(1):last_date
                start_time = DateTime(day) + Hour(22)
                end_time = DateTime(day) + Day(1) + Minute(45) + Hour(21)

                subset = filter(row -> start_time <= row.datetime <= end_time, df)
                daily_vals = clamp.(Float64.(subset[!, col_sym]), lower_bound, upper_bound)
                daily_vals_norm = scaler(daily_vals)
                push!(forecasters_dict[name], daily_vals_norm)
            end

        end

        ############ Extract True Values ############
        df_ecmwf = DataFrame(Parquet2.Dataset(model_paths["ecmwf_xgb"]))
        df_ecmwf = dropmissing(df_ecmwf)
        df_ecmwf[!, :measured] = clamp.(df_ecmwf[!, :measured], lower_bound, upper_bound)

        first_date = Date(first(df_ecmwf.datetime))
        last_date = Date(last(df_ecmwf.datetime)) - Day(1)
        for day in first_date:Day(1):last_date
            start_time = DateTime(day) + Hour(22)
            end_time = DateTime(day) + Day(1) + Minute(45) + Hour(21)

            daily_data = filter(row -> start_time <= row.datetime <= end_time, df_ecmwf)
            daily_data = clamp.(Float64.(daily_data[!, :measured]), lower_bound, upper_bound)
            push!(true_values, daily_data)
            push!(dates, day)
        end

        return true_values, forecasters_dict, scaler, dates
    end

end
