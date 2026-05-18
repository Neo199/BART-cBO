# ---------------------------------------------------------
# SEMIDEFINITE PROGRAMMING - Min
# ---------------------------------------------------------

sdp_relaxation <- function(alpha, n_vars, lambda, removed_columns) {

# Append missing variables back with zero coefficients
alpha_vect <- c(alpha, setNames(rep(0, length(removed_columns)), removed_columns))

# Reorder so that variables are in a consistent order
alpha_vect <- alpha_vect[order(names(alpha_vect))]

# Extract coefficients
b <- alpha_vect[2:(n_vars + 1)] + lambda
a <- alpha_vect[(n_vars + 2):length(alpha_vect)]

# Create index for upper triangle (i < j)
idx_prod <- t(combn(n_vars, 2))
n_idx <- nrow(idx_prod)

if (length(a) != n_idx) {
  stop("Number of Coefficients does not match the number of off-diagonal terms!")
}

  # Convert a to symmetric matrix A
  A <- matrix(0, n_vars, n_vars)
  for (i in 1:n_idx) {
    A[idx_prod[i, 1], idx_prod[i, 2]] <- a[i] / 2
    A[idx_prod[i, 2], idx_prod[i, 1]] <- a[i] / 2
  }
# browser()
  # Construct the lifted matrix At (like in SDP)
  bt <- b / 2 + A %*% rep(1, n_vars) / 2
  At <- rbind(
    cbind(A / 4, bt / 2),
    c(t(bt) / 2, 0)
  )

  # Define the variable X as a positive semidefinite matrix
  X <- Variable(n_vars + 1, n_vars + 1, PSD = TRUE)
  
  # Define the objective function
  obj <- Minimize(matrix_trace(At %*% X))
  
  # Define the constraints
  constraints <- list(
    diag(X) == rep(1, n_vars + 1)
  )
  
  prob <- Problem(obj, constraints)
  result <- solve(prob, solver = "SCS")  # or use "MOSEK" if available
  
  # Extract vectors and compute Cholesky
  # Add small identity matrix if X.value is numerically not PSD
  X_value <- result$getValue(X)
  
  project_to_psd <- function(M) {
    eig <- eigen(M, symmetric = TRUE)
    eig$values[eig$values < 0] <- 1e-9   # clamp to small positive
    M_psd <- eig$vectors %*% diag(eig$values) %*% t(eig$vectors)
    return(M_psd)
  }
  
  X_psd <- project_to_psd(X_value)
  L <- chol(X_psd)
  L <- t(L)   # convert to lower triangular
  

  # Repeat rounding for different vectors
  n_rand_vector <- 100
  model_vect <- matrix(0, nrow = n_vars, ncol = n_rand_vector)
  obj_vect <- numeric(n_rand_vector)
  
  for (kk in 1:n_rand_vector) {
    # Generate a random cutting plane vector (uniformly
    # distributed on the unit sphere - normalized vector)
    r <- rnorm(n_vars+1)
    r <- r / sqrt(sum(r^2))  # normalize
    
    y_soln <- sign(t(L) %*% r)
    
    # Convert solution to original domain and assign to output vector
    model_vect[, kk] <- (y_soln[1:n_vars] + 1) / 2
    obj_vect[kk] <- t(model_vect[, kk]) %*% A %*% model_vect[, kk] + 
      sum(b * model_vect[, kk])
  }
  
  # Find optimal rounded solution
  opt_idx <- which.min(obj_vect)
  model <- model_vect[, opt_idx]
  obj <- obj_vect[opt_idx]

  return(list(model = model, obj = obj))
}


