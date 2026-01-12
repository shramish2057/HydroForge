# HydroForge Default Configuration
# Default parameter values for simulation

# Default simulation parameters
const DEFAULT_DT_MAX = 1.0          # Maximum timestep (s)
const DEFAULT_CFL = 0.7             # CFL number
const DEFAULT_H_MIN = 0.001         # Minimum depth for wet cell (m)
const DEFAULT_G = 9.81              # Gravity (m/s²)
const DEFAULT_OUTPUT_INTERVAL = 60.0 # Output frequency (s)

# Default Manning's roughness coefficients
const MANNING_IMPERVIOUS = 0.03     # Streets, concrete
const MANNING_PERVIOUS = 0.05       # Grass, soil
const MANNING_WATER = 0.025         # Open water

# Numerical tolerances
const DEPTH_TOL = 1e-10             # Depth tolerance for dry cells
const MASS_BALANCE_TOL = 0.01       # Acceptable mass balance error (1%)
