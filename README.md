# HydroForge

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Julia](https://img.shields.io/badge/Julia-1.10%2B-9558B2.svg)](https://julialang.org/)
[![Status](https://img.shields.io/badge/Status-Pre--Alpha-orange.svg)]()

**Real-Time Urban Flood Risk Simulator**

*Empowering cities to simulate, visualize, and mitigate urban flooding in real time – before the rain even falls.*

---

## Overview

HydroForge is an open-source platform built in Julia that delivers real-time hydrodynamic simulation of urban stormwater runoff, drainage system performance, and surface inundation. Leveraging Julia's unparalleled speed for numerical computing, it enables interactive, scenario-based flood risk assessment.

### Key Features

- **Blazing-fast simulations** – True real-time capability using efficient shallow water equation solvers
- **Fully open-source** – No vendor lock-in, MIT licensed
- **Urban drainage focus** – Designed for pluvial flood prediction in cities
- **High societal impact** – Supports proactive flood mitigation in vulnerable communities

## Quick Start

### Prerequisites

- Julia 1.10 or higher
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/hydroforge/HydroForge.git
cd HydroForge

# Install dependencies
julia --project -e 'using Pkg; Pkg.instantiate()'
```

### Run the Demo

```julia
using HydroForge

# Display version information
HydroForge.info()

# Run the bundled demo scenario
HydroForge.run_demo()
```

### Run a Custom Scenario

```julia
using HydroForge

# Run your own scenario
HydroForge.run("path/to/your/scenario.toml")
```

## Project Status

**Current: Pre-Alpha (v0.1.0-dev)**

This project is in early development. Core solver functionality is being implemented and tested.

### MVP Scope (In Progress)
- [x] Core data types (Grid, State, Parameters)
- [x] Local inertial 2D solver
- [x] Rainfall source term
- [ ] Demo dataset generation
- [ ] CLI interface
- [ ] HTTP API
- [ ] Web UI

### Future Roadmap
- 1D drainage network coupling
- GPU acceleration (CUDA.jl)
- Probabilistic forecasting
- Multi-user cloud deployment

## Architecture

```
HydroForge/
├── src/
│   ├── types/        # Core data structures
│   ├── numerics/     # Solver algorithms
│   ├── physics/      # Physical processes
│   ├── io/           # Input/output
│   ├── models/       # Simulation runners
│   ├── api/          # HTTP API (planned)
│   └── cli/          # Command-line interface
├── test/             # Test suite
├── docs/             # Documentation
├── assets/           # Demo data
└── examples/         # Example scenarios
```

## Technical Approach

HydroForge uses the **local inertial approximation** of the shallow water equations, which provides:

- Numerical stability without complex Riemann solvers
- Semi-implicit friction treatment for robustness
- Efficient explicit time-stepping with CFL control
- Accurate representation of subcritical urban flows

### Key Equations

**Continuity:**
```
∂h/∂t + ∂qx/∂x + ∂qy/∂y = R
```

**Momentum (local inertial):**
```
∂q/∂t + gh∂η/∂x + gn²|q|q/h^(10/3) = 0
```

Where:
- `h` = water depth
- `q` = unit discharge
- `η` = water surface elevation
- `R` = rainfall rate
- `n` = Manning's roughness
- `g` = gravity

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone and setup
git clone https://github.com/hydroforge/HydroForge.git
cd HydroForge
julia --project -e 'using Pkg; Pkg.instantiate()'

# Run tests
julia --project -e 'using Pkg; Pkg.test()'

# Format code (requires JuliaFormatter)
julia --project -e 'using JuliaFormatter; format("src")'
```

### Good First Issues

Looking to contribute? Check our issues labeled `good first issue`.

## Documentation

Full documentation coming soon. For now:

- **API Reference**: See docstrings in `src/`
- **Examples**: See `examples/` directory
- **Architecture**: See this README

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use HydroForge in your research, please cite:

```bibtex
@software{hydroforge2026,
  title = {HydroForge: Real-Time Urban Flood Risk Simulator},
  year = {2026},
  url = {https://github.com/hydroforge/HydroForge}
}
```

## Acknowledgments

HydroForge is inspired by and builds upon:

- [ShallowWaters.jl](https://github.com/milankl/ShallowWaters.jl) – Efficient shallow water solver
- [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) – Julia's scientific computing ecosystem
- The broader Julia community

---

**HydroForge** – Making flood prediction accessible to everyone.
