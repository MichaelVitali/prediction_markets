module RealWorldtestData

using DataFrames
using Parquet2
using Dates
using Normalization
using CSV

export preprocessing_forecasts

    function preprocessing_forecasts(model_paths::Dict{String, String}, q)

        forecasters_dict = Dict{String, Vector{Vector{Float64}}}()
        scalers = Dict{String, MinMax}()
        true_values = []

        for name in keys(model_paths)
            forecasters_dict[name] = []
        end

        ######## Extract MODELS FORECASTS ########
        col_sym = Symbol("q$(Int(q * 100))")        
        for (name, path) in model_paths
            df = DataFrame(Parquet2.Dataset(path))
            df = dropmissing(df)

            first_date = Date(first(df.datetime))
            last_date = Date(last(df.datetime)) - Day(1)
            scaler = MinMax(df[!, col_sym])
            scalers[name] = scaler

            for day in first_date:Day(1):last_date
                start_time = DateTime(day) + Hour(22)
                end_time = DateTime(day) + Day(1) + Minute(45) + Hour(21)

                subset = filter(row -> start_time <= row.datetime <= end_time, df)
                daily_vals = Float64.(subset[!, col_sym])
                daily_vals_norm = scaler(daily_vals)
                push!(forecasters_dict[name], daily_vals_norm)
            end

        end

        ############ Extract True Values ############
        df_ecmwf = DataFrame(Parquet2.Dataset(model_paths["ecmwf_xgb"]))
        df_ecmwf = dropmissing(df_ecmwf)

        first_date = Date(first(df_ecmwf.datetime))
        last_date = Date(last(df_ecmwf.datetime)) - Day(1)
        scaler_target = MinMax(df_ecmwf[!, :measured])

        for day in first_date:Day(1):last_date
            start_time = DateTime(day) + Hour(22)
            end_time = DateTime(day) + Day(1) + Minute(45) + Hour(21)

            daily_data = filter(row -> start_time <= row.datetime <= end_time, df_ecmwf)
            daily_data = daily_data[!, :measured]
            push!(true_values, daily_data)
        end

        return true_values, forecasters_dict, scalers, scaler_target
    end

end