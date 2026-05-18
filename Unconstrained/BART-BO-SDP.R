# -------------------------------------------------------------
# SDP-on-EI Bayesian Optimization (replace GA)
# -------------------------------------------------------------
# Requires: BART, CVXR, Matrix
library(BART)
library(CVXR)
library(Matrix)
library(GA)
# ---------------------------
# Your adapted SDP function
# ---------------------------
sdp_relaxation <- function(alpha, n_vars, lambda = 0, removed_columns = integer(0),
                           n_rand_vector = 200, solver = "SCS") {
  # alpha: c(intercept, b1..bn, a_offdiag in combn order) and objective minimized is:
  #    obj(x) = x' A x + b' x  (note intercept ignored for argmin)
  # We'll assume the user has prepared alpha already for the optimization they want to perform.
  alpha_vect <- c(alpha, setNames(rep(0, length(removed_columns)), removed_columns))
  if (!is.null(names(alpha_vect))) alpha_vect <- alpha_vect[order(names(alpha_vect))]
  
  b <- alpha_vect[2:(n_vars + 1)] + lambda
  a <- if (length(alpha_vect) >= (n_vars + 2)) alpha_vect[(n_vars + 2):length(alpha_vect)] else numeric(0)
  
  idx_prod <- t(combn(n_vars, 2))
  n_idx <- nrow(idx_prod)
  if (length(a) != n_idx && n_idx > 0) stop("Number of coefficients does not match off-diagonal terms!")
  
  A <- matrix(0, n_vars, n_vars)
  if (n_idx > 0) {
    for (i in 1:n_idx) {
      A[idx_prod[i,1], idx_prod[i,2]] <- a[i] / 2
      A[idx_prod[i,2], idx_prod[i,1]] <- a[i] / 2
    }
  }
  
  # Construct At (lifted matrix as in your original function)
  bt <- b / 2 + A %*% rep(1, n_vars) / 2
  At <- rbind(
    cbind(A / 4, bt / 2),
    c(t(bt) / 2, 0)
  )
  
  # CVXR SDP
  X <- Variable(n_vars + 1, n_vars + 1, PSD = TRUE)
  obj <- Minimize(matrix_trace(At %*% X))
  constraints <- list(diag(X) == rep(1, n_vars + 1))
  prob <- Problem(obj, constraints)
  result <- solve(prob, solver = solver)
  X_value <- result$getValue(X)
  
  # Project numerically to PSD
  symm <- (X_value + t(X_value)) / 2
  ev <- eigen(symm, symmetric = TRUE)
  ev$values[ev$values < 1e-9] <- 1e-9
  X_psd <- ev$vectors %*% diag(ev$values) %*% t(ev$vectors)
  
  # get sqrt via eigen
  ev2 <- eigen(X_psd, symmetric = TRUE)
  sqrtX <- ev2$vectors %*% diag(sqrt(pmax(ev2$values, 0))) %*% t(ev2$vectors)
  # use chol of sqrtX to generate L such that X_psd ≈ L %*% t(L)
  L <- tryCatch({
    chol(X_psd)
  }, error = function(e) {
    # fallback: use sqrtX's chol
    chol(sqrtX + diag(1e-9, nrow(sqrtX)))
  })
  L <- t(L)
  
  # random hyperplane rounding
  model_vect <- matrix(0, nrow = n_vars, ncol = n_rand_vector)
  obj_vect <- numeric(n_rand_vector)
  for (kk in 1:n_rand_vector) {
    r <- rnorm(n_vars + 1)
    r <- r / sqrt(sum(r^2))
    y_soln <- sign(t(L) %*% r)
    x_bin <- (y_soln[1:n_vars] + 1) / 2
    model_vect[, kk] <- x_bin
    obj_vect[kk] <- as.numeric(t(x_bin) %*% A %*% x_bin + sum(b * x_bin))
  }
  opt_idx <- which.min(obj_vect)
  model <- as.numeric(model_vect[, opt_idx])
  obj <- obj_vect[opt_idx]
  return(list(model = model, obj = obj, X = X_value))
}

# ---------------------------
# EI calculation (BART posterior samples -> EI)
# ---------------------------
ei_from_bart <- function(bart_fit, x_vec, f_best) {
  # x_vec: numeric binary vector length p
  # returns EI scalar
  preds <- predict(bart_fit, newdata = matrix(x_vec, nrow = 1))
  # preds returned: ndpost x 1 (or 1 x ndpost) — coerce to vector
  preds_v <- as.numeric(preds)
  mu <- mean(preds_v)
  sigma <- sd(preds_v)
  if (sigma == 0) return(0)
  z <- (mu - f_best) / sigma
  ei <- (mu - f_best) * pnorm(z) + sigma * dnorm(z)
  if (ei < 0) ei <- 0
  return(ei)
}

# ---------------------------
# Fit a quadratic surrogate to EI(x)
# ---------------------------
fit_quadratic_to_EI <- function(bart_fit, p, f_best, n_samples = 800) {
  # Sample random binary points
  X_try <- matrix(rbinom(n_samples * p, 1, 0.5), nrow = n_samples, ncol = p)
  ei_vec <- numeric(n_samples)
  for (i in 1:n_samples) {
    ei_vec[i] <- ei_from_bart(bart_fit, X_try[i, ], f_best)
  }
  # Build design matrix: intercept + main + pairwise
  pair_idx <- t(combn(p, 2))
  design <- cbind(1, X_try)
  for (k in 1:nrow(pair_idx)) {
    design <- cbind(design, X_try[, pair_idx[k,1]] * X_try[, pair_idx[k,2]])
  }
  # Linear model: EI ≈ design %*% coef
  lm_fit <- lm(ei_vec ~ 0 + design)  # design includes intercept col
  coefs <- coef(lm_fit)
  # Safety: if some coefs are NA (perfect collinearity), replace by 0
  coefs[is.na(coefs)] <- 0
  intercept <- as.numeric(coefs[1])
  b <- as.numeric(coefs[2:(p + 1)])
  a_off <- if (length(coefs) > (p + 1)) as.numeric(coefs[(p + 2):length(coefs)]) else numeric(0)
  return(list(intercept = intercept, b = b, a_off = a_off))
}

# ---------------------------
# Toy problem & initial data (your original problem)
# ---------------------------
set.seed(1)
p <- 8
n_init <- 10
X <- matrix(rbinom(n_init * p, 1, 0.5), nrow = n_init, ncol = p)

true_f <- function(x) {
  stopifnot(length(x) == 8)
  linear_part <- sum(c(1.2, -0.8, 0.5, 1.0, -1.5, 0.3, 0.7, -0.4) * x)
  interaction_part <- 0.8 * x[1] * x[2] - 0.6 * x[3] * x[4] + 0.4 * x[5] * x[6]
  nonlinear_part <- 0.5 * sin(pi * sum(x) / 8)
  return(linear_part + interaction_part + nonlinear_part)
}
y <- apply(X, 1, true_f)

# Fit BART
bart_fit <- wbart(x.train = X, y.train = y, ndpost = 1000, nskip = 200, ntree = 50)
f_best <- max(y)

# ---------------------------
# BO loop: replace GA by SDP-on-EI
# ---------------------------
n_iterations <- 20
for (iter in 1:n_iterations) {
  cat("Iteration", iter, "- data size", nrow(X), "f_best", f_best, "\n")
  # 1) Fit quadratic surrogate directly to EI(x)
  quad_ei <- fit_quadratic_to_EI(bart_fit, p, f_best, n_samples = 700)
  
  # 2) Build alpha vector: intercept, b1..bp, a_offdiag
  alpha_ei <- c(quad_ei$intercept, quad_ei$b, quad_ei$a_off)
  # We want to *maximize* EI(x). sdp_relaxation minimizes x' A x + b' x,
  # so negate the linear and quadratic coefficients to convert max -> min.
  alpha_for_sdp <- alpha_ei
  alpha_for_sdp[2:(p+1)] <- -alpha_ei[2:(p+1)]
  if (length(alpha_ei) > (p + 1)) {
    alpha_for_sdp[(p+2):length(alpha_for_sdp)] <- -alpha_ei[(p+2):length(alpha_ei)]
  }
  
  # 3) Call SDP relaxation to get candidate that (approximately) maximizes EI
  sdp_out <- sdp_relaxation(alpha_for_sdp, n_vars = p, lambda = 1e-6, n_rand_vector = 300)
  x_next <- as.numeric(sdp_out$model)
  cat("SDP proposed x_next:", x_next, "\n")
  
  # 4) Evaluate true f, update dataset and BART
  y_next <- true_f(x_next)
  X <- rbind(X, x_next)
  y <- c(y, y_next)
  f_best <- max(f_best, y_next)
  bart_fit <- wbart(x.train = X, y.train = y, ndpost = 1000, nskip = 200, ntree = 50)
  cat("Observed y_next:", y_next, " new f_best:", f_best, "\n\n")
}

# ---------------------------
# Final: report best found
# ---------------------------
best_idx <- which.max(y)
cat("Best observed y:", y[best_idx], " at x:", X[best_idx,], "\n")

# -------------------------------------------------------------
# GA
# -------------------------------------------------------------

GA_run <- ga(type = "binary", fitness = true_f, nBits = p,
             popSize = 100, maxiter = 1000, run = 100, monitor = FALSE)
ga_result <- list(solution = GA_run@solution, fitness_value = GA_run@fitnessValue)
ga_result$solution
