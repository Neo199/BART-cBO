# -------------------------------------------------------------
# Comparison: DART vs BART vs GA for Binary Quadratic Programming
# Three-way comparison with consistent evaluation counting
# -------------------------------------------------------------
options(java.parameters = "-Xmx2g")
Sys.setenv(JAVA_TOOL_OPTIONS = "-XX:ParallelGCThreads=1")
library(dartMachine)
library(BART)
library(GA)

set.seed(1)

# Problem setup
p <- 100
alpha <- 1
lambda <- 1e-4

# Generate Q matrix
quad_mat <- function(n_vars, alpha) {
  K <- function(s, t) exp(-1 * (s - t)^2 / alpha)
  decay <- matrix(0, n_vars, n_vars)
  for (i in 1:n_vars) {
    for (j in 1:n_vars) {
      decay[i, j] <- K(i, j)
    }
  }
  Q <- matrix(rnorm(n_vars * n_vars), n_vars, n_vars)
  return(Q * decay)
}

Q <- quad_mat(p, alpha)
eval_counter <- 0

true_f <- function(x) {
  eval_counter <<- eval_counter + 1
  as.numeric(t(as.numeric(x)) %*% Q %*% as.numeric(x))
}

true_f_batch <- function(X_mat) apply(X_mat, 1, true_f)

normalize_01 <- function(x) {
  rng <- range(x)
  if (diff(rng) == 0) return(rep(0.5, length(x)))
  (x - rng[1]) / diff(rng)
}

n_init <- 20
n_iter <- 250

cat("========================================\n")
cat("BINARY QUADRATIC PROGRAMMING COMPARISON\n")
cat("========================================\n")
cat("Problem:", p, "variables\n")
cat("Budget:", n_init, "init +", n_iter, "iters =", n_init + n_iter, "total\n\n")

# ========================================
# METHOD 1: DART-BO (dartMachine)
# ========================================
cat("METHOD 1: DART-BO (dartMachine)\n")
cat("----------------------------------------\n")

eval_counter <- 0
X_dart <- matrix(rbinom(n_init * p, 1, 0.5), n_init, p)
y_dart <- true_f_batch(X_dart)
X_dart_df <- data.frame(X_dart)
y_dart_norm <- normalize_01(y_dart)
dart_fit <- bartMachine(X_dart_df, y_dart_norm, num_trees = 100)
f_best_dart <- min(y_dart)
best_idx <- which.min(y_dart)
x_best   <- X_dart[best_idx, ]
cat("Init: best =", round(f_best_dart, 4), "\n")

for (iter in 1:n_iter) {
  fitness_dart <- function(x) {
    x_in <- as.data.frame(matrix(as.numeric(x), 1))
    colnames(x_in) <- colnames(X_dart_df)
    post <- bart_machine_get_posterior(dart_fit, x_in)
    pred <- as.numeric(post$y_hat_posterior_samples)
    mu <- mean(pred, na.rm = TRUE)
    sigma <- sd(pred, na.rm = TRUE)
    if (is.na(mu)) mu <- 0
    if (is.na(sigma) || sigma < 1e-10) sigma <- 1e-6
    f_norm <- (f_best_dart - min(y_dart)) / (max(y_dart) - min(y_dart))
    z <- (f_norm - mu) / sigma
    max((f_norm - mu) * pnorm(z) + sigma * dnorm(z), 0)
  }
  
  GA_res <- ga(type = "binary", nBits = p, fitness = fitness_dart,
               popSize = 200, maxiter = 40, run = 20, keepBest = TRUE,
               suggestions = matrix(x_best, nrow = 1))
  
  x_next <- as.numeric(GA_res@solution[1, ])
  y_next <- true_f(x_next)
  X_dart <- rbind(X_dart, x_next)
  y_dart <- c(y_dart, y_next)
  # Update best
  if (y_next < f_best_dart) {
    f_best_dart <- y_next
    x_best      <- x_next
  }
  X_dart_df <- data.frame(X_dart)
  y_dart_norm <- normalize_01(y_dart)
  f_best_dart <- min(y_dart)
  dart_fit <- bartMachine(X_dart_df, y_dart_norm, num_trees = 100)
  cat(sprintf("Iter %3d: best = %8.4f\n", iter, f_best_dart))
  
}

dart_evals <- eval_counter
cat("DART Final: evals =", dart_evals, ", best =", round(f_best_dart, 6), "\n\n")

# ========================================
# METHOD 2: BART-BO (BART package)
# ========================================
cat("METHOD 2: BART-BO (BART package)\n")
cat("----------------------------------------\n")

eval_counter <- 0
X_bart <- matrix(rbinom(n_init * p, 1, 0.5), n_init, p)
y_bart <- true_f_batch(X_bart)
y_bart_norm <- normalize_01(y_bart)
bart_fit <- wbart(X_bart, y_bart_norm, ndpost=1000, nskip=200, ntree=100, printevery=100000)
f_best_bart <- min(y_bart)
best_idx <- which.min(y_bart)
x_best   <- X_dart[best_idx, ]

cat("Init: best =", round(f_best_bart, 4), "\n")

for (iter in 1:n_iter) {
  fitness_bart <- function(x) {
    x_mat <- matrix(as.numeric(x), 1)
    pred <- predict(bart_fit, x_mat)
    if (nrow(pred) < ncol(pred)) pred <- t(pred)
    mu <- mean(pred)
    sigma <- sd(pred)
    if (is.na(mu)) mu <- 0
    if (is.na(sigma) || sigma < 1e-10) sigma <- 1e-6
    f_norm <- (f_best_bart - min(y_bart)) / (max(y_bart) - min(y_bart))
    z <- (f_norm - mu) / sigma
    max((f_norm - mu) * pnorm(z) + sigma * dnorm(z), 0)
  }
  
  GA_res <- ga(type = "binary", nBits = p, fitness = fitness_bart,
               popSize = 200, maxiter = 40, run = 20, keepBest = TRUE, 
               suggestions = matrix(x_best, nrow = 1))

  x_next <- as.numeric(GA_res@solution[1, ])
  y_next <- true_f(x_next)
  X_bart <- rbind(X_bart, x_next)
  y_bart <- c(y_bart, y_next)
  # Update best
  if (y_next < f_best_bart) {
    f_best_bart <- y_next
    x_best      <- x_next
  }
  y_bart_norm <- normalize_01(y_bart)
  f_best_bart <- min(y_bart)
  bart_fit <- wbart(X_bart, y_bart_norm, ndpost=1000, nskip=200, ntree=100, printevery=100000)
  
  if (iter %% 20 == 0 || iter == 1) {
    cat(sprintf("Iter %3d: best = %8.4f\n", iter, f_best_bart))
  }
}

bart_evals <- eval_counter
cat("BART Final: evals =", bart_evals, ", best =", round(f_best_bart, 6), "\n\n")

# ========================================
# METHOD 3: Pure GA Benchmark
# ========================================
cat("METHOD 3: PURE GA BENCHMARK\n")
cat("----------------------------------------\n")

eval_counter <- 0
GA_final <- ga(type = "binary", fitness = function(x) -true_f(x), nBits = p,
               popSize = 200, maxiter = 200, run = 100, monitor = FALSE)
ga_evals <- eval_counter
ga_best <- -GA_final@fitnessValue

cat("GA Final: evals =", ga_evals, ", best =", round(ga_best, 6), "\n\n")

# ========================================
# FINAL COMPARISON
# ========================================
cat("\n========================================\n")
cat("FINAL COMPARISON SUMMARY\n")
cat("========================================\n\n")

results <- data.frame(
  Method = c("DART-BO", "BART-BO", "Pure GA"),
  Evaluations = c(dart_evals, bart_evals, ga_evals),
  Best_Objective = round(c(f_best_dart, f_best_bart, ga_best), 6),
  Eval_Efficiency = sprintf("%.1f%%", 100 * c(dart_evals, bart_evals, ga_evals) / ga_evals)
)
print(results)

cat("\n--- Performance Analysis ---\n\n")

best_obj <- min(c(f_best_dart, f_best_bart, ga_best))
best_method <- c("DART-BO", "BART-BO", "Pure GA")[which.min(c(f_best_dart, f_best_bart, ga_best))]

cat(sprintf("★ Best solution: %.6f by %s\n\n", best_obj, best_method))

# DART vs GA
if (f_best_dart < ga_best) {
  impr <- (ga_best - f_best_dart) / abs(ga_best) * 100
  cat(sprintf("DART-BO: %.2f%% better than GA, using %.1f%% of evaluations\n", 
              impr, 100 * dart_evals / ga_evals))
} else {
  gap <- (f_best_dart - ga_best) / abs(ga_best) * 100
  cat(sprintf("DART-BO: %.2f%% worse than GA, but %.1fx fewer evaluations\n", 
              gap, ga_evals / dart_evals))
}

# BART vs GA
if (f_best_bart < ga_best) {
  impr <- (ga_best - f_best_bart) / abs(ga_best) * 100
  cat(sprintf("BART-BO: %.2f%% better than GA, using %.1f%% of evaluations\n", 
              impr, 100 * bart_evals / ga_evals))
} else {
  gap <- (f_best_bart - ga_best) / abs(ga_best) * 100
  cat(sprintf("BART-BO: %.2f%% worse than GA, but %.1fx fewer evaluations\n", 
              gap, ga_evals / bart_evals))
}

# DART vs BART
cat("\nDART vs BART:\n")
if (abs(f_best_dart - f_best_bart) / min(abs(f_best_dart), abs(f_best_bart)) < 0.01) {
  cat("  Similar performance\n")
} else if (f_best_dart < f_best_bart) {
  impr <- (f_best_bart - f_best_dart) / abs(f_best_bart) * 100
  cat(sprintf("  DART %.2f%% better\n", impr))
} else {
  impr <- (f_best_dart - f_best_bart) / abs(f_best_dart) * 100
  cat(sprintf("  BART %.2f%% better\n", impr))
}

cat("\n========================================\n")
cat("Avg BO evals:", round(mean(c(dart_evals, bart_evals))), 
    sprintf("(%.1f%% of GA)", 100 * mean(c(dart_evals, bart_evals)) / ga_evals), "\n")
cat("========================================\n")