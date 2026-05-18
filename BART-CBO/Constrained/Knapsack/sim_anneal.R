# ---------------------------------------------------------
# SIMULATED ANNEALING
# ---------------------------------------------------------

simulated_annealing <- function(objective, n_vars, evalBudget, x_current) {
  
  # Initialize matrices to store solutions
  model_iter <- matrix(0, nrow = evalBudget, ncol = n_vars)
  obj_iter <- numeric(evalBudget)
  
  # Initial temperature and cooling schedule
  T <- 1.0
  cool <- function(T) 0.8 * T
  
  # Generate initial solution and evaluate objective
  old_x <- x_current
  old_obj <- objective(old_x)
  
  # Initialize best solution
  best_x <- old_x
  best_obj <- old_obj
  
  # Run simulated annealing
  for (t in 1:evalBudget) {
    # Decrease temperature
    T <- cool(T)
    
    # Generate new candidate by flipping a random bit
    flip_bit <- sample(1:n_vars, 1)
    new_x <- old_x
    new_x[flip_bit] <- 1 - new_x[flip_bit]
    
    # Evaluate objective function
    new_obj <- objective(new_x)
    
    # Accept new solution if it improves or probabilistically accept it
    if ((new_obj < old_obj) || (runif(1) < exp((old_obj - new_obj) / T))) {
      old_x <- new_x
      old_obj <- new_obj
    }
    
    # Update best solution
    if (new_obj < best_obj) {
      best_x <- new_x
      best_obj <- new_obj
    }
    
    # Save solution
    model_iter[t, ] <- best_x
    obj_iter[t] <- best_obj
  }
  
  return(list(model_iter = model_iter, obj_iter = obj_iter))
}