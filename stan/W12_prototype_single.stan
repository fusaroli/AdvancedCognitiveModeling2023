
// Stan Code for Prototype Model using Kalman Filter (W12_prototype_single.stan)

data {
  int<lower=1> ntrials;                // Number of trials
  int<lower=1> nfeatures;              // Number of feature dimensions (e.g., 2)
  array[ntrials] int<lower=0, upper=1> cat_one;  // True category labels (0 or 1) provided as feedback
  array[ntrials] int<lower=0, upper=1> y;        // Observed participant decisions (0 or 1)
  array[ntrials, nfeatures] real obs;  // Stimulus features (trials x features)
  
  // Priors / Fixed Values
  real<lower=0, upper=1> b;            // Response bias (e.g., 0.5 for no bias)
  vector[nfeatures] initial_mu_cat0;   // Initial mean vector for prototype 0
  vector[nfeatures] initial_mu_cat1;   // Initial mean vector for prototype 1
  real<lower=0> initial_sigma_diag;    // Initial diagonal value for covariance matrices
  
  // Prior parameters for log_r
  real prior_logr_mean;
  real<lower=0> prior_logr_sd;
}

parameters {
  // Parameter to estimate: Observation noise variance on log scale
  real log_r;                          
}

transformed parameters {
  // Transform log_r to observation noise variance (r_value)
  // Using exp() ensures positivity. Add a small epsilon for stability if needed.
  real<lower=1e-6> r_value = exp(log_r); 
  
  // Store response probabilities for each trial
  array[ntrials] real<lower=0.0001, upper=0.9999> p; // Prob of choosing category 1
  
  // --- Kalman Filter Simulation within Stan ---
  // Initialize prototypes (means and covariance matrices)
  vector[nfeatures] mu_cat0 = initial_mu_cat0;
  vector[nfeatures] mu_cat1 = initial_mu_cat1;
  matrix[nfeatures, nfeatures] sigma_cat0 = diag_matrix(rep_vector(initial_sigma_diag, nfeatures));
  matrix[nfeatures, nfeatures] sigma_cat1 = diag_matrix(rep_vector(initial_sigma_diag, nfeatures));
  
  // Observation noise matrix (diagonal)
  matrix[nfeatures, nfeatures] r_matrix = diag_matrix(rep_vector(r_value, nfeatures));
  matrix[nfeatures, nfeatures] I = diag_matrix(rep_vector(1.0, nfeatures)); // Identity matrix

  // Process trials sequentially 
  for (i in 1:ntrials) {
    vector[nfeatures] current_obs = to_vector(obs[i]);
    
    // --- Categorization Decision Probability ---
    matrix[nfeatures, nfeatures] cov_cat0 = sigma_cat0 + r_matrix;
    matrix[nfeatures, nfeatures] cov_cat1 = sigma_cat1 + r_matrix;
    
    // Calculate log probability densities using multi_normal_lpdf for robustness
    // Add bias term b (log(b) for cat 1, log(1-b) for cat 0)
    real log_p0 = multi_normal_lpdf(current_obs | mu_cat0, cov_cat0) + log(1 - b);
    real log_p1 = multi_normal_lpdf(current_obs | mu_cat1, cov_cat1) + log(b);
    
    // Calculate probability of choosing category 1 using log_sum_exp
    real log_sum_p = log_sum_exp(log_p0, log_p1);
    real prob_cat_1 = exp(log_p1 - log_sum_p);
    
    // Bound probabilities away from 0 and 1 for numerical stability
    p[i] = fmax(fmin(prob_cat_1, 0.9999), 0.0001);

    // --- Learning Update (based on true feedback cat_one[i]) ---
    // Only update if it's not the very last observation 
    // (or adjust loop if feedback influences next trial's decision directly)
    // This logic assumes decision P is based on state *before* update for trial i.
    if (i < ntrials) { // Optional: only update if there are future trials
      if (cat_one[i] == 1) { // Update prototype 1
        vector[nfeatures] innovation = current_obs - mu_cat1;
        matrix[nfeatures, nfeatures] S = sigma_cat1 + r_matrix;
        // Use mdivide_right_spd for K = Sigma * S^-1 (more stable)
        matrix[nfeatures, nfeatures] K = mdivide_right_spd(sigma_cat1, S); 
        mu_cat1 = mu_cat1 + K * innovation;
        matrix[nfeatures, nfeatures] IK = I - K;
        // Joseph form update for sigma
        sigma_cat1 = IK * sigma_cat1 * IK' + K * r_matrix * K'; 
      } else { // Update prototype 0
        vector[nfeatures] innovation = current_obs - mu_cat0;
        matrix[nfeatures, nfeatures] S = sigma_cat0 + r_matrix;
        matrix[nfeatures, nfeatures] K = mdivide_right_spd(sigma_cat0, S);
        mu_cat0 = mu_cat0 + K * innovation;
        matrix[nfeatures, nfeatures] IK = I - K;
        sigma_cat0 = IK * sigma_cat0 * IK' + K * r_matrix * K';
      }
       // Ensure symmetry after update (optional but good practice)
       sigma_cat0 = 0.5 * (sigma_cat0 + sigma_cat0');
       sigma_cat1 = 0.5 * (sigma_cat1 + sigma_cat1');
    }
  }
}

model {
  // Prior for the observation noise parameter (on log scale)
  target += normal_lpdf(log_r | prior_logr_mean, prior_logr_sd);  
  
  // Likelihood: Observed decisions y follow a Bernoulli distribution
  // with probabilities p calculated in transformed parameters
  target += bernoulli_lpmf(y | p);
}

generated quantities {
  // Calculate log likelihood for each trial for model comparison (e.g., LOO)
  array[ntrials] real log_lik;
  for (i in 1:ntrials) {
    log_lik[i] = bernoulli_lpmf(y[i] | p[i]);
  }
  
  // Posterior predictive checks: simulate data based on estimated parameters
  array[ntrials] int y_pred;
  for (i in 1:ntrials) {
    y_pred[i] = bernoulli_rng(p[i]);
  }

  // Save estimated r_value on original scale
  real estimated_r_value = r_value; 
}
