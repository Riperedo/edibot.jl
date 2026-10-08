# test/runtests.jl

using Test
using EdiBot
using Random
using LinearAlgebra
using Statistics

@testset "EdiBot.jl Test Suite" begin

    @testset "Mathematical Benchmark Optimizations" begin
        # 1. Sphere Function: f(x) = sum(x.^2), global minimum at x = [0, 0, 0], f* = 0
        sphere(p) = sum(p .^ 2)
        lb = [-5.0, -5.0, -5.0]
        ub = [5.0, 5.0, 5.0]
        
        # Test PSO on Sphere
        res_pso = optimize(sphere; lower_bounds=lb, upper_bounds=ub, algorithm=PSO(), options=OptimizationOptions(max_iters=200, pop_size=40, seed=42))
        @test res_pso.best_cost < 1e-4
        @test all(abs.(res_pso.best_params) .< 0.05)
        
        # Test DE on Sphere
        res_de = optimize(sphere; lower_bounds=lb, upper_bounds=ub, algorithm=DE(), options=OptimizationOptions(max_iters=200, pop_size=40, seed=42))
        @test res_de.best_cost < 1e-4
        @test all(abs.(res_de.best_params) .< 0.05)

        # Test GA on Sphere
        res_ga = optimize(sphere; lower_bounds=lb, upper_bounds=ub, algorithm=GA(), options=OptimizationOptions(max_iters=300, pop_size=60, seed=42))
        @test res_ga.best_cost < 1e-2

        # 2. 2D Rosenbrock Function: f(x, y) = (1 - x)^2 + 100*(y - x^2)^2, min at (1, 1), f* = 0
        rosenbrock(p) = (1.0 - p[1])^2 + 100.0 * (p[2] - p[1]^2)^2
        res_rosen_pso = optimize(rosenbrock; lower_bounds=[-2.0, -2.0], upper_bounds=[2.0, 2.0], algorithm=PSO(), options=OptimizationOptions(max_iters=400, pop_size=80, seed=123))
        @test res_rosen_pso.best_cost < 1e-2
        @test isapprox(res_rosen_pso.best_params[1], 1.0, atol=0.1)
        @test isapprox(res_rosen_pso.best_params[2], 1.0, atol=0.1)
    end

    @testset "Physical Model Fitting: KWW & VFT" begin
        # Synthetic KWW data: A=1.0, tau=5.0, beta=0.6, y0=0.05
        true_kww_params = [1.0, 5.0, 0.6, 0.05]
        t_data = collect(range(0.1, 25.0, length=50))
        y_kww_clean = [kww(t, true_kww_params) for t in t_data]
        
        # Fit KWW with PSO
        kww_fit = fit(
            :kww,
            t_data,
            y_kww_clean;
            lower_bounds = [0.5, 1.0, 0.2, 0.0],
            upper_bounds = [1.5, 15.0, 1.0, 0.2],
            algorithm = PSO(),
            options = OptimizationOptions(max_iters=300, pop_size=50, seed=777)
        )
        
        @test kww_fit.metrics[:r2] > 0.999
        @test isapprox(kww_fit.opt_result.best_params[1], 1.0, atol=0.05) # A
        @test isapprox(kww_fit.opt_result.best_params[2], 5.0, atol=0.1)  # tau
        @test isapprox(kww_fit.opt_result.best_params[3], 0.6, atol=0.05) # beta

        # Synthetic VFT Log10 data: log10(tau0)=-12.0, B=1200.0, T0=150.0
        true_vft_params = [-12.0, 1200.0, 150.0]
        T_data = collect(range(170.0, 350.0, length=40))
        y_vft = [vft_log10(T, true_vft_params) for T in T_data]

        vft_fit = fit(
            :vft_log10,
            T_data,
            y_vft;
            lower_bounds = [-15.0, 500.0, 100.0],
            upper_bounds = [-9.0, 2500.0, 165.0],
            algorithm = DE(),
            options = OptimizationOptions(max_iters=300, pop_size=50, seed=42)
        )

        @test vft_fit.metrics[:r2] > 0.999
        @test isapprox(vft_fit.opt_result.best_params[3], 150.0, atol=2.0) # T0
    end

    @testset "Physical Model Fitting: Hard Sphere S(q)" begin
        # Generate Percus-Yevick S(q) data: phi=0.35, sigma=1.0, scale=1.0
        true_hs_params = [0.35, 1.0, 1.0]
        q_data = collect(range(0.2, 20.0, length=80))
        sq_clean = [hard_sphere_sq(q, true_hs_params) for q in q_data]

        hs_fit = fit(
            :hard_sphere_sq,
            q_data,
            sq_clean;
            lower_bounds = [0.1, 0.5, 0.8],
            upper_bounds = [0.55, 1.5, 1.2],
            algorithm = PSO(),
            options = OptimizationOptions(max_iters=250, pop_size=50, seed=123)
        )

        @test hs_fit.metrics[:r2] > 0.999
        @test isapprox(hs_fit.opt_result.best_params[1], 0.35, atol=0.02) # phi
        @test isapprox(hs_fit.opt_result.best_params[2], 1.0, atol=0.05)  # sigma
    end

    @testset "Analytical String Expression Parser" begin
        # Test expression: "a * exp(-b * x) + c"
        x_pts = collect(range(0.0, 5.0, length=30))
        y_pts = [2.5 * exp(-1.2 * x) + 0.3 for x in x_pts]
        
        expr_fit = fit(
            "a * exp(-b * x) + c",
            x_pts,
            y_pts;
            lower_bounds = [1.0, 0.1, 0.0],
            upper_bounds = [5.0, 3.0, 1.0],
            algorithm = DE(),
            options = OptimizationOptions(max_iters=200, pop_size=40, seed=42)
        )
        
        @test expr_fit.metrics[:r2] > 0.999
        @test isapprox(expr_fit.opt_result.best_params[1], 2.5, atol=0.05)
        @test isapprox(expr_fit.opt_result.best_params[2], 1.2, atol=0.05)
        @test isapprox(expr_fit.opt_result.best_params[3], 0.3, atol=0.05)
    end

    @testset "Loss Functions & Statistical Metrics" begin
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 1.9, 3.2, 3.8, 5.0]
        
        @test MSE()(y_true, y_pred) ≈ (0.01 + 0.01 + 0.04 + 0.04 + 0.0) / 5
        @test RMSE()(y_true, y_pred) ≈ sqrt(MSE()(y_true, y_pred))
        @test MAE()(y_true, y_pred) ≈ (0.1 + 0.1 + 0.2 + 0.2 + 0.0) / 5
        
        w = [1.0, 2.0, 1.0, 2.0, 1.0]
        wloss = WeightedMSE(w)
        @test wloss(y_true, y_pred) > 0.0
        
        logloss = LogMSE()
        @test logloss(y_true, y_pred) > 0.0
        
        metrics = evaluate_metrics(y_true, y_pred, 2)
        @test metrics[:r2] > 0.95
        @test metrics[:rmse] ≈ RMSE()(y_true, y_pred)
    end

    @testset "I/O & Report Exporters" begin
        tmp_dir = mktempdir()
        data_file = joinpath(tmp_dir, "test_data.dat")
        
        # Write temporary dataset
        open(data_file, "w") do io
            println(io, "# Test Dataset Header")
            for x in 1:10
                println(io, x, "  ", 2.0 * x + 1.0)
            end
        end
        
        # Load dataset
        x_loaded, y_loaded, _ = load_dataset(data_file)
        @test length(x_loaded) == 10
        @test x_loaded[1] ≈ 1.0
        @test y_loaded[end] ≈ 21.0
        
        # Fit linear model
        fit_res = fit(
            "m * x + b",
            x_loaded,
            y_loaded;
            lower_bounds = [0.0, 0.0],
            upper_bounds = [5.0, 5.0],
            algorithm = PSO(),
            options = OptimizationOptions(max_iters=100, seed=1)
        )
        
        # Test JSON exporter
        json_file = joinpath(tmp_dir, "result.json")
        export_json(json_file, fit_res)
        @test isfile(json_file)
        @test filesize(json_file) > 50
        
        # Test LaTeX exporter
        tex_file = joinpath(tmp_dir, "report.tex")
        export_latex_report(tex_file, fit_res; dataset_name="TestLinear")
        @test isfile(tex_file)
        @test contains(read(tex_file, String), "\\begin{table}")
        
        # Clean up
        rm(tmp_dir, recursive=true)
    end
end
