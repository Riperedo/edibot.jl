# EdiBot.jl 🤖
### High-Performance Numerical Optimizer & Analytical Curve Fitting Engine in Julia

[![Build Status](https://github.com/Riperedo/edibot.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/Riperedo/edibot.jl/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**EdiBot.jl** is a generic, high-performance numerical optimization and curve-fitting framework in pure Julia. Designed for physics, chemistry, materials science, and numerical engineering, EdiBot offers global heuristic search engines (**Particle Swarm Optimization**, **Differential Evolution**, and **Genetic Algorithms**) with native multi-threading and zero external C dependencies.

---

## 🌟 Key Features

- ⚡ **Pure Julia Performance:** Native multi-threaded fitness evaluations (`Threads.@threads`) with zero compilation overhead.
- 🧬 **Multiple Global Optimizers:**
  - **Particle Swarm Optimization (`PSO`)**: Adaptive inertia decay, cognitive/social swarm mechanics, velocity clamping.
  - **Differential Evolution (`DE`)**: Mutation schemes (`:rand1bin`, `:best1bin`, `:current_to_best1bin`) with crossover rate $CR$.
  - **Real-Coded Genetic Algorithm (`GA`)**: Simulated Binary Crossover (SBX), polynomial mutation, tournament selection, and elitism.
- 🔬 **Built-in Physical & Mathematical Models:**
  - **Vogel-Fulcher-Tammann (VFT):** Standard $\tau(T)$ and $\log_{10}\tau(T)$ formulations for glass physics and viscosity.
  - **Kohlrausch-Williams-Watts (KWW):** Stretched exponential relaxation $\phi(t) = A \exp(-(t/\tau)^\beta) + y_0$.
  - **Static Structure Factors $S(q)$:** Analytical Percus-Yevick Hard Sphere and Screened Coulomb / Yukawa RPA (integrates seamlessly with [AnalyticalStructureFactors.jl](https://github.com/Riperedo/AnalyticalStructureFactors.jl)).
  - **Standard Functions:** Gaussian, Lorentzian, Power Law, Exponential Decay, Sigmoid, Polynomials.
- 🔤 **Dynamic Analytical Expression Parser:** Write arbitrary mathematical strings (e.g. `"A * exp(-(x / tau)^beta) + y0"`) and compile them on the fly.
- 🎯 **Flexible Loss Functions:** `MSE`, `RMSE`, `MAE`, `LogMSE` (for data spanning decades/orders of magnitude), and `WeightedMSE` (with low-$q$ or experimental variance weights).
- 📊 **Automated Scientific Reporting:** Automatic computation of $R^2$, Adjusted $R^2$, RMSE, MAE, AIC, BIC, residuals, JSON output, and formatted publication-ready LaTeX tables.
- 🖥️ **Command Line Interface (CLI):** Ready-to-use executable `bin/edibot` for scripting, automated pipelines, or shell usage.

---

## 📦 Installation

Clone the repository and instantiate the project:

```bash
git clone https://github.com/Riperedo/edibot.jl.git
cd edibot.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

---

## 🚀 Quickstart: Julia API

### 1. Fitting Physical Models (VFT & KWW)

```julia
using EdiBot

# Fit Vogel-Fulcher-Tammann (VFT) relaxation data
T_data = [200.0, 210.0, 220.0, 230.0, 250.0, 280.0, 320.0, 380.0]
log_tau = [20.08, 9.22, 3.79, 0.53, -3.19, -5.98, -7.85, -9.22]

result = fit(
    :vft_log10,
    T_data,
    log_tau;
    lower_bounds = [-15.0, 500.0, 100.0],
    upper_bounds = [-10.0, 3000.0, 195.0],
    algorithm = DE()
)

println("Best parameters: ", result.opt_result.best_params)
println("R² goodness of fit: ", result.metrics[:r2])
```

### 2. Fitting Custom Julia Functions

```julia
using EdiBot

# Define custom scalar function (x, p) -> y
double_decay(t, p) = p[1] * exp(-t / p[2]) + p[3] * exp(-t / p[4])

t = collect(range(0.1, 10.0, length=50))
y = [double_decay(ti, [1.5, 0.8, 0.5, 4.0]) for ti in t]

fit_res = fit(
    double_decay,
    t,
    y;
    lower_bounds = [0.0, 0.1, 0.0, 1.0],
    upper_bounds = [3.0, 2.0, 2.0, 10.0],
    algorithm = PSO(),
    options = OptimizationOptions(max_iters=300, pop_size=60)
)
```

### 3. Fitting Analytical Formula Strings

```julia
using EdiBot

x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
y = [2.5 * exp(-1.2 * xi) + 0.3 for xi in x]

res = fit(
    "a * exp(-b * x) + c",
    x,
    y;
    lower_bounds = [1.0, 0.1, 0.0],
    upper_bounds = [5.0, 3.0, 1.0],
    algorithm = PSO()
)
```

---

## 🖥️ Command Line Interface (CLI)

EdiBot includes a standalone CLI launcher located at `bin/edibot`:

```bash
# Fit KWW relaxation data using Particle Swarm Optimization (PSO)
julia --project bin/edibot --data examples/data/kww_sample.dat \
  --expr "A * exp(-(x / tau)^beta)" \
  --algo pso \
  --bounds "A:0.5..1.5, tau:0.5..10.0, beta:0.2..1.0" \
  --out-json results.json \
  --out-report report.tex

# Fit Hard-Sphere S(q) static structure factor data
julia --project bin/edibot --data examples/data/hs_sample.dat \
  --model hs \
  --algo de \
  --bounds "0.1..0.6, 0.5..1.5, 0.8..1.2" \
  --out-report hs_report.tex
```

---

## 🧪 Running Tests

To run the full unit test suite:

```bash
julia --project -e 'using Pkg; Pkg.test()'
```
Or directly:
```bash
julia --project test/runtests.jl
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
