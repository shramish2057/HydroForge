# HydroForge Evaporation Module
# Evapotranspiration (ET) modeling for water surface and land

# =============================================================================
# Evaporation Parameters
# =============================================================================

"""
    EvaporationParameters{T<:AbstractFloat}

Parameters for evaporation modeling.

# Fields
- `method::Symbol`: ET calculation method (:constant, :penman, :priestley_taylor, :hargreaves)
- `rate::T`: Constant evaporation rate (m/s) - used when method=:constant
- `albedo::T`: Surface albedo (0-1), default 0.23 for water
- `elevation::T`: Site elevation (m) for pressure calculations
- `latitude::T`: Site latitude (degrees) for radiation calculations

# Notes
For short flood simulations (<24h), evaporation is typically negligible.
For longer simulations or water balance studies, evaporation becomes important.
"""
struct EvaporationParameters{T<:AbstractFloat}
    method::Symbol
    rate::T              # Constant rate (m/s)
    albedo::T            # Surface albedo
    elevation::T         # Site elevation (m)
    latitude::T          # Site latitude (degrees)
end

"""
    EvaporationParameters(; kwargs...)

Create evaporation parameters.

# Keyword Arguments
- `method`: Calculation method (default: :constant)
  - `:constant`: Use fixed rate
  - `:penman`: Penman-Monteith (requires meteorological data)
  - `:priestley_taylor`: Simplified energy balance
  - `:hargreaves`: Temperature-based empirical method
- `rate`: Constant evaporation rate in mm/day (default: 5.0)
- `albedo`: Surface albedo (default: 0.23)
- `elevation`: Site elevation in m (default: 0.0)
- `latitude`: Site latitude in degrees (default: 45.0)
- `T`: Float type (default: Float64)
"""
function EvaporationParameters(;
    method::Symbol=:constant,
    rate::Real=5.0,         # mm/day
    albedo::Real=0.23,
    elevation::Real=0.0,
    latitude::Real=45.0,
    T::Type{<:AbstractFloat}=Float64
)
    # Convert mm/day to m/s
    rate_ms = T(rate / (1000.0 * 86400.0))

    EvaporationParameters{T}(method, rate_ms, T(albedo), T(elevation), T(latitude))
end

"""
    EvaporationParameters(rate_mm_day::Real; T=Float64)

Create simple constant evaporation parameters.
"""
function EvaporationParameters(rate_mm_day::Real; T::Type{<:AbstractFloat}=Float64)
    EvaporationParameters(method=:constant, rate=rate_mm_day, T=T)
end

# =============================================================================
# Meteorological Data for Evaporation
# =============================================================================

"""
    MeteorologicalData{T<:AbstractFloat}

Meteorological data for ET calculations.

# Fields
- `temperature::T`: Air temperature (°C)
- `humidity::T`: Relative humidity (0-1)
- `wind_speed::T`: Wind speed at 2m (m/s)
- `solar_radiation::T`: Solar radiation (W/m²)
- `day_of_year::Int`: Day of year (1-365)
"""
struct MeteorologicalData{T<:AbstractFloat}
    temperature::T
    humidity::T
    wind_speed::T
    solar_radiation::T
    day_of_year::Int
end

"""
    MeteorologicalData(; kwargs...)

Create meteorological data with defaults.
"""
function MeteorologicalData(;
    temperature::Real=20.0,
    humidity::Real=0.6,
    wind_speed::Real=2.0,
    solar_radiation::Real=300.0,
    day_of_year::Int=180,
    T::Type{<:AbstractFloat}=Float64
)
    MeteorologicalData{T}(T(temperature), T(humidity), T(wind_speed),
                          T(solar_radiation), day_of_year)
end

# =============================================================================
# Time-Varying Evaporation
# =============================================================================

"""
    EvaporationTimeSeries{T<:AbstractFloat}

Time-varying evaporation rates.

# Fields
- `times::Vector{T}`: Time points (s)
- `rates::Vector{T}`: Evaporation rates at each time (m/s)
"""
struct EvaporationTimeSeries{T<:AbstractFloat}
    times::Vector{T}
    rates::Vector{T}  # m/s
end

"""
    EvaporationTimeSeries(times, rates_mm_day; T=Float64)

Create evaporation time series from mm/day rates.
"""
function EvaporationTimeSeries(times::Vector{<:Real}, rates_mm_day::Vector{<:Real};
                                T::Type{<:AbstractFloat}=Float64)
    @assert length(times) == length(rates_mm_day) "Times and rates must have same length"
    @assert issorted(times) "Times must be sorted"

    # Convert mm/day to m/s
    conversion = T(1.0 / (1000.0 * 86400.0))
    rates_ms = T.(rates_mm_day) .* conversion

    EvaporationTimeSeries{T}(T.(times), rates_ms)
end

"""
    evaporation_rate(et::EvaporationTimeSeries, t::Real)

Get interpolated evaporation rate at time t (m/s).
"""
function evaporation_rate(et::EvaporationTimeSeries{T}, t::Real) where T
    times = et.times
    rates = et.rates

    if t <= times[1]
        return rates[1]
    elseif t >= times[end]
        return rates[end]
    end

    # Linear interpolation
    idx = searchsortedlast(times, T(t))
    t1, t2 = times[idx], times[idx+1]
    r1, r2 = rates[idx], rates[idx+1]
    α = (T(t) - t1) / (t2 - t1)
    return (one(T) - α) * r1 + α * r2
end

# =============================================================================
# Evaporation Calculations
# =============================================================================

"""
    saturation_vapor_pressure(T_celsius)

Calculate saturation vapor pressure (kPa) using Tetens formula.
"""
function saturation_vapor_pressure(T_celsius::T) where T<:AbstractFloat
    # Tetens formula
    T(0.6108) * exp(T(17.27) * T_celsius / (T_celsius + T(237.3)))
end

"""
    psychrometric_constant(elevation)

Calculate psychrometric constant γ (kPa/°C) at given elevation.
"""
function psychrometric_constant(elevation::T) where T<:AbstractFloat
    # Atmospheric pressure at elevation
    P = T(101.3) * ((T(293.0) - T(0.0065) * elevation) / T(293.0))^T(5.26)
    # Psychrometric constant
    T(0.665e-3) * P
end

"""
    slope_saturation_curve(T_celsius)

Calculate slope of saturation vapor pressure curve (kPa/°C).
"""
function slope_saturation_curve(T_celsius::T) where T<:AbstractFloat
    es = saturation_vapor_pressure(T_celsius)
    T(4098.0) * es / (T_celsius + T(237.3))^2
end

"""
    penman_monteith_et(meteo, params)

Calculate reference ET using FAO Penman-Monteith method.
Returns evaporation rate in m/s.
"""
function penman_monteith_et(meteo::MeteorologicalData{T},
                            params::EvaporationParameters{T}) where T
    Temp = meteo.temperature
    RH = meteo.humidity
    u2 = meteo.wind_speed
    Rs = meteo.solar_radiation

    # Vapor pressure calculations
    es = saturation_vapor_pressure(Temp)
    ea = es * RH
    vpd = es - ea  # Vapor pressure deficit

    # Psychrometric constant
    γ = psychrometric_constant(params.elevation)

    # Slope of saturation curve
    Δ = slope_saturation_curve(Temp)

    # Net radiation (simplified)
    α = params.albedo
    Rn = (one(T) - α) * Rs * T(0.0036)  # Convert W/m² to MJ/m²/hr approximately

    # Soil heat flux (assume small for hourly)
    G = zero(T)

    # FAO-56 Penman-Monteith equation (mm/day equivalent, then convert)
    numerator = T(0.408) * Δ * (Rn - G) + γ * T(900.0) / (Temp + T(273.0)) * u2 * vpd
    denominator = Δ + γ * (one(T) + T(0.34) * u2)

    ET0_mm_day = numerator / denominator

    # Convert mm/day to m/s
    ET0_mm_day / T(1000.0 * 86400.0)
end

"""
    priestley_taylor_et(meteo, params; α_pt=1.26)

Calculate ET using Priestley-Taylor method (radiation-based).
Returns evaporation rate in m/s.
"""
function priestley_taylor_et(meteo::MeteorologicalData{T},
                             params::EvaporationParameters{T};
                             α_pt::T=T(1.26)) where T
    Temp = meteo.temperature
    Rs = meteo.solar_radiation

    # Psychrometric constant
    γ = psychrometric_constant(params.elevation)

    # Slope of saturation curve
    Δ = slope_saturation_curve(Temp)

    # Net radiation (simplified)
    α = params.albedo
    Rn = (one(T) - α) * Rs * T(0.0036)

    # Priestley-Taylor equation
    ET = α_pt * (Δ / (Δ + γ)) * Rn / T(2.45)  # mm/day approximately

    # Convert to m/s
    max(zero(T), ET / T(1000.0 * 86400.0))
end

"""
    hargreaves_et(T_min, T_max, T_mean, Ra; day_of_year=180, latitude=45.0)

Calculate ET using Hargreaves method (temperature-based).
Returns evaporation rate in m/s.

# Arguments
- `T_min`: Minimum daily temperature (°C)
- `T_max`: Maximum daily temperature (°C)
- `T_mean`: Mean daily temperature (°C)
- `Ra`: Extraterrestrial radiation (MJ/m²/day)
"""
function hargreaves_et(T_min::T, T_max::T, T_mean::T, Ra::T) where T<:AbstractFloat
    # Hargreaves equation
    ET = T(0.0023) * (T_mean + T(17.8)) * sqrt(max(T_max - T_min, zero(T))) * Ra

    # Convert mm/day to m/s
    max(zero(T), ET / T(1000.0 * 86400.0))
end

"""
    calculate_et_rate(params, meteo=nothing)

Calculate evaporation rate based on method and data.
Returns rate in m/s.
"""
function calculate_et_rate(params::EvaporationParameters{T},
                           meteo::Union{MeteorologicalData{T},Nothing}=nothing) where T
    if params.method == :constant
        return params.rate
    elseif params.method == :penman && meteo !== nothing
        return penman_monteith_et(meteo, params)
    elseif params.method == :priestley_taylor && meteo !== nothing
        return priestley_taylor_et(meteo, params)
    else
        # Fall back to constant rate
        return params.rate
    end
end

# =============================================================================
# Evaporation Application
# =============================================================================

"""
    apply_evaporation!(h::Matrix, params::EvaporationParameters, dt::Real)

Apply evaporation losses to water depths.

# Arguments
- `h`: Water depth matrix (modified in-place)
- `params`: Evaporation parameters
- `dt`: Timestep (s)

# Returns
- Total evaporated depth (m, sum over all wet cells)
"""
function apply_evaporation!(h::Matrix{T}, params::EvaporationParameters{T},
                            dt::Real) where T
    rate = params.rate
    evap_depth = rate * T(dt)
    total_evaporated = zero(T)

    @inbounds for j in axes(h, 2), i in axes(h, 1)
        if h[i, j] > zero(T)
            actual = min(evap_depth, h[i, j])
            h[i, j] -= actual
            total_evaporated += actual
        end
    end

    total_evaporated
end

"""
    apply_evaporation!(h::Matrix, et::EvaporationTimeSeries, t::Real, dt::Real)

Apply time-varying evaporation losses.
"""
function apply_evaporation!(h::Matrix{T}, et::EvaporationTimeSeries{T},
                            t::Real, dt::Real) where T
    rate = evaporation_rate(et, t)
    evap_depth = rate * T(dt)
    total_evaporated = zero(T)

    @inbounds for j in axes(h, 2), i in axes(h, 1)
        if h[i, j] > zero(T)
            actual = min(evap_depth, h[i, j])
            h[i, j] -= actual
            total_evaporated += actual
        end
    end

    total_evaporated
end

"""
    apply_evaporation!(h::Matrix, params::EvaporationParameters, meteo::MeteorologicalData, dt::Real)

Apply evaporation using meteorological data for ET calculation.
"""
function apply_evaporation!(h::Matrix{T}, params::EvaporationParameters{T},
                            meteo::MeteorologicalData{T}, dt::Real) where T
    rate = calculate_et_rate(params, meteo)
    evap_depth = rate * T(dt)
    total_evaporated = zero(T)

    @inbounds for j in axes(h, 2), i in axes(h, 1)
        if h[i, j] > zero(T)
            actual = min(evap_depth, h[i, j])
            h[i, j] -= actual
            total_evaporated += actual
        end
    end

    total_evaporated
end

# =============================================================================
# Evaporation Analysis
# =============================================================================

"""
    daily_evaporation(params::EvaporationParameters)

Calculate daily evaporation depth (mm).
"""
function daily_evaporation(params::EvaporationParameters{T}) where T
    # Convert m/s to mm/day
    params.rate * T(1000.0 * 86400.0)
end

"""
    cumulative_evaporation(rate_ms, duration_s)

Calculate cumulative evaporation depth (mm).
"""
function cumulative_evaporation(rate_ms::T, duration_s::Real) where T<:AbstractFloat
    rate_ms * T(duration_s) * T(1000.0)  # m to mm
end

"""
    pan_to_lake_evaporation(pan_evap_mm; pan_coefficient=0.7)

Convert pan evaporation to lake/open water evaporation.
"""
function pan_to_lake_evaporation(pan_evap_mm::T; pan_coefficient::T=T(0.7)) where T<:AbstractFloat
    pan_evap_mm * pan_coefficient
end
