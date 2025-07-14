using Pkg
Pkg.activate(@__DIR__)
Pkg.resolve()
Pkg.instantiate()

using OptimalGIV
using DataFrames
using CSV
using JSON
using StatsModels
using CategoricalArrays
using LinearAlgebra
using OptimalGIV.HeteroPCA: StandardHeteroPCA, DiagonalDeletion, DeflatedHeteroPCA

# Function to extract model results into a serializable format
function extract_model_results(model)
    results = Dict{String, Any}()
    
    # Basic coefficients and vcov
    results["endog_coef"] = Vector(model.endog_coef)
    results["exog_coef"] = Vector(model.exog_coef)
    results["endog_vcov"] = Matrix(model.endog_vcov)
    results["exog_vcov"] = Matrix(model.exog_vcov)
    
    # Aggregate coefficient
    if isa(model.agg_coef, Number)
        results["agg_coef"] = model.agg_coef
    else
        results["agg_coef"] = Vector(model.agg_coef)
    end
    
    # Model metadata
    results["complete_coverage"] = model.complete_coverage
    results["formula"] = string(model.formula)
    results["formula_schema"] = string(model.formula_schema)
    # residual_variance could be a scalar, vector, or matrix
    if isa(model.residual_variance, Number)
        results["residual_variance"] = model.residual_variance
    elseif isa(model.residual_variance, Vector)
        results["residual_variance"] = Vector(model.residual_variance)
    else
        results["residual_variance"] = Matrix(model.residual_variance)
    end
    results["responsename"] = string(model.responsename)
    results["endogname"] = string(model.endogname)
    results["endog_coefnames"] = [string(n) for n in model.endog_coefnames]
    results["exog_coefnames"] = [string(n) for n in model.exog_coefnames]
    results["idvar"] = string(model.idvar)
    results["tvar"] = string(model.tvar)
    results["weightvar"] = isnothing(model.weightvar) ? nothing : string(model.weightvar)
    
    # Exclude pairs
    exclude_dict = Dict{String, Vector{Any}}()
    for (k, v) in model.exclude_pairs
        exclude_dict[string(k)] = collect(v)
    end
    results["exclude_pairs"] = exclude_dict
    
    # Model statistics
    results["converged"] = model.converged
    results["N"] = model.N
    results["T"] = model.T
    results["nobs"] = model.nobs
    results["dof"] = model.dof
    results["dof_residual"] = model.dof_residual
    
    # PC results if applicable
    results["n_pcs"] = model.n_pcs
    if model.n_pcs > 0
        results["pc_factors"] = Matrix(model.pc_factors)
        results["pc_loadings"] = Matrix(model.pc_loadings)
    end
    
    return results
end

# Function to save test case
function save_test_case(name::String, parameters::Dict, results::Dict)
    test_case = Dict(
        "model_name" => name,
        "parameters" => parameters,
        "results" => results
    )
    
    filename = joinpath(@__DIR__, "outputs", "$(name).json")
    open(filename, "w") do f
        JSON.print(f, test_case, 4)
    end
    
    println("Saved test case: $name")
end

# Create output directory
mkpath(joinpath(@__DIR__, "outputs"))

# Load the test data
df = CSV.read(joinpath(@__DIR__, "..", "..", "examples", "simdata1.csv"), DataFrame)
df.id = categorical(df.id)

println("Running Julia test cases...")

# Test 1: Homogeneous elasticity with scalar search
println("\n1. Homogeneous elasticity - scalar search")
params = Dict(
    "formula" => "q + endog(p) ~ 0 + fe(id) & (η1 + η2)",
    "algorithm" => "scalar_search",
    "guess" => Dict("Aggregate" => 2.0)
)
model = giv(df, @formula(q + endog(p) ~ 0 + fe(id) & (η1 + η2)), 
            :id, :t, :absS;
            algorithm=:scalar_search,
            guess=Dict("Aggregate" => 2.0),
            save=:all)
save_test_case("homogeneous_scalar_search", params, extract_model_results(model))

# Test 2: Homogeneous elasticity with IV
println("\n2. Homogeneous elasticity - IV")
params = Dict(
    "formula" => "q + endog(p) ~ 0 + fe(id) & (η1 + η2)",
    "algorithm" => "iv",
    "guess" => 1.0
)
model = giv(df, @formula(q + endog(p) ~ 0 + fe(id) & (η1 + η2)), 
            :id, :t, :absS;
            algorithm=:iv,
            guess=1.0)
save_test_case("homogeneous_iv", params, extract_model_results(model))

# Test 3: Heterogeneous elasticity with scalar search
println("\n3. Heterogeneous elasticity - scalar search")
params = Dict(
    "formula" => "q + id & endog(p) ~ 0 + id & (η1 + η2)",
    "algorithm" => "scalar_search",
    "guess" => Dict("Aggregate" => 2.5)
)
model = giv(df, @formula(q + id & endog(p) ~ 0 + id & (η1 + η2)), 
            :id, :t, :absS;
            algorithm=:scalar_search,
            guess=Dict("Aggregate" => 2.5))
save_test_case("heterogeneous_scalar_search", params, extract_model_results(model))

# Test 4: Heterogeneous elasticity with IV
println("\n4. Heterogeneous elasticity - IV")
params = Dict(
    "formula" => "q + id & endog(p) ~ 0 + id & (η1 + η2)",
    "algorithm" => "iv",
    "guess" => [1.0, 1.0, 1.0, 1.0, 1.0]
)
model = giv(df, @formula(q + id & endog(p) ~ 0 + id & (η1 + η2)), 
            :id, :t, :absS;
            algorithm=:iv,
            guess=ones(5))
save_test_case("heterogeneous_iv", params, extract_model_results(model))

# Test 5: Debiased OLS
println("\n5. Homogeneous elasticity - debiased OLS")
params = Dict(
    "formula" => "q + endog(p) ~ 0 + fe(id) & (η1 + η2)",
    "algorithm" => "debiased_ols",
    "guess" => 1.0
)
model = giv(df, @formula(q + endog(p) ~ 0 + fe(id) & (η1 + η2)), 
            :id, :t, :absS;
            algorithm=:debiased_ols,
            guess=1.0)
save_test_case("homogeneous_debiased_ols", params, extract_model_results(model))

# Test 6: Two-pass IV
println("\n6. Homogeneous elasticity - two-pass IV")
params = Dict(
    "formula" => "q + endog(p) ~ 0 + fe(id) & (η1 + η2)",
    "algorithm" => "iv_twopass",
    "guess" => 1.0
)
model = giv(df, @formula(q + endog(p) ~ 0 + fe(id) & (η1 + η2)), 
            :id, :t, :absS;
            algorithm=:iv_twopass,
            guess=1.0)
save_test_case("homogeneous_iv_twopass", params, extract_model_results(model))

# Test 7: Partial absorption
println("\n7. Partial absorption")
params = Dict(
    "formula" => "q + endog(p) ~ 0 + id & η1 + fe(id) & η2",
    "algorithm" => "scalar_search",
    "guess" => Dict("Aggregate" => 2.0)
)
model = giv(df, @formula(q + endog(p) ~ 0 + id & η1 + fe(id) & η2), 
            :id, :t, :absS;
            algorithm=:scalar_search,
            guess=Dict("Aggregate" => 2.0))
save_test_case("partial_absorption", params, extract_model_results(model))

# Test 8: With exclude pairs
println("\n8. With exclude pairs")
params = Dict(
    "formula" => "q + endog(p) ~ 0 + fe(id) & (η1 + η2)",
    "algorithm" => "iv",
    "guess" => 1.0,
    "exclude_pairs" => Dict("1" => [2, 3], "4" => [5])
)
model = giv(df, @formula(q + endog(p) ~ 0 + fe(id) & (η1 + η2)), 
            :id, :t, :absS;
            algorithm=:iv,
            guess=1.0,
            exclude_pairs=Dict(1 => [2, 3], 4 => [5]))
save_test_case("with_exclude_pairs", params, extract_model_results(model))

# Test 9: With PC extraction (2 factors)
println("\n9. With PC extraction")
params = Dict(
    "formula" => "q + endog(p) ~ fe(id) + pc(2)",
    "algorithm" => "iv",
    "guess" => 1.0
)
model = giv(df, @formula(q + endog(p) ~ fe(id) + pc(2)), 
            :id, :t, :absS;
            algorithm=:iv,
            guess=1.0)
save_test_case("with_pc_extraction", params, extract_model_results(model))

# Test 10: With solver options
println("\n10. With solver options")
params = Dict(
    "formula" => "q + endog(p) ~ 0 + fe(id) & (η1 + η2)",
    "algorithm" => "iv",
    "guess" => 1.0,
    "solver_options" => Dict("iterations" => 1)
)
model = giv(df, @formula(q + endog(p) ~ 0 + fe(id) & (η1 + η2)), 
            :id, :t, :absS;
            algorithm=:iv,
            guess=1.0,
            solver_options=(iterations=1,))
save_test_case("with_solver_options", params, extract_model_results(model))

# Test 11: PCA standard pairwise
println("\n11. PCA standard pairwise")
params = Dict(
    "formula" => "q + endog(p) ~ 0 + pc(2)",
    "algorithm" => "iv",
    "guess" => Dict("p" => 2.0),
    "pca_option" => Dict(
        "impute_method" => "pairwise",
        "algorithm" => "StandardHeteroPCA",
        "demean" => true,
        "maxiter" => 100,
        "abstol" => 1e-7
    ),
    "save" => "all",
    "save_df" => true,
    "quiet" => true
)
model = giv(df, @formula(q + endog(p) ~ 0 + pc(2)), 
            :id, :t, :absS;
            algorithm=:iv,
            guess=Dict(:p => 2.0),
            pca_option=(
                impute_method=:pairwise,
                algorithm=StandardHeteroPCA(),
                demean=true,
                maxiter=100,
                abstol=1e-7
            ),
            save=:all,
            save_df=true,
            quiet=true)
results = extract_model_results(model)
# Add PCA-specific results
pc_model = model.pc_model
results["pc_mean"] = pc_model.mean
results["pc_projection"] = Matrix(pc_model.proj')
results["pc_prinvars"] = pc_model.prinvars
results["pc_noisevars"] = pc_model.noisevars
results["pc_r2"] = OptimalGIV.HeteroPCA.r2(pc_model)
results["pc_converged"] = pc_model.converged
results["pc_iterations"] = pc_model.iterations
save_test_case("pca_standard_pairwise", params, results)

# Test 12: PCA diagonal zero
println("\n12. PCA diagonal zero")
params = Dict(
    "formula" => "q + endog(p) ~ 0 + pc(2)",
    "algorithm" => "iv",
    "guess" => Dict("p" => 2.0),
    "pca_option" => Dict(
        "impute_method" => "zero",
        "algorithm" => "DiagonalDeletion",
        "demean" => false
    ),
    "save" => "all",
    "save_df" => true,
    "quiet" => true
)
model = giv(df, @formula(q + endog(p) ~ 0 + pc(2)), 
            :id, :t, :absS;
            algorithm=:iv,
            guess=Dict(:p => 2.0),
            pca_option=(
                impute_method=:zero,
                algorithm=DiagonalDeletion(),
                demean=false
            ),
            save=:all,
            save_df=true,
            quiet=true)
results = extract_model_results(model)
# Add PCA-specific results
pc_model = model.pc_model
results["pc_mean"] = pc_model.mean
results["pc_projection"] = Matrix(pc_model.proj')
results["pc_prinvars"] = pc_model.prinvars
results["pc_noisevars"] = pc_model.noisevars
results["pc_r2"] = OptimalGIV.HeteroPCA.r2(pc_model)
results["pc_converged"] = pc_model.converged
results["pc_iterations"] = pc_model.iterations
save_test_case("pca_diagonal_zero", params, results)

# Test 13: PCA deflated pairwise
println("\n13. PCA deflated pairwise")
params = Dict(
    "formula" => "q + endog(p) ~ 0 + pc(2)",
    "algorithm" => "iv",
    "guess" => Dict("p" => 2.0),
    "pca_option" => Dict(
        "impute_method" => "pairwise",
        "algorithm" => "DeflatedHeteroPCA",
        "algorithm_options" => Dict(
            "t_block" => 5,
            "condition_number_threshold" => 3.5
        ),
        "demean" => false,
        "α" => 1.0,
        "suppress_warnings" => false,
        "abstol" => 1e-6
    ),
    "save" => "all",
    "save_df" => true,
    "quiet" => true
)
model = giv(df, @formula(q + endog(p) ~ 0 + pc(2)), 
            :id, :t, :absS;
            algorithm=:iv,
            guess=Dict(:p => 2.0),
            pca_option=(
                impute_method=:pairwise,
                algorithm=DeflatedHeteroPCA(t_block=5, condition_number_threshold=3.5),
                demean=false,
                α=1.0,
                suppress_warnings=false,
                abstol=1e-6
            ),
            save=:all,
            save_df=true,
            quiet=true)
results = extract_model_results(model)
# Add PCA-specific results
pc_model = model.pc_model
results["pc_mean"] = pc_model.mean
results["pc_projection"] = Matrix(pc_model.proj')
results["pc_prinvars"] = pc_model.prinvars
results["pc_noisevars"] = pc_model.noisevars
results["pc_r2"] = OptimalGIV.HeteroPCA.r2(pc_model)
results["pc_converged"] = pc_model.converged
results["pc_iterations"] = pc_model.iterations
save_test_case("pca_deflated_pairwise", params, results)

println("\nAll Julia test cases completed!")
println("Output files saved in: ", joinpath(@__DIR__, "outputs"))