
// Multilevel Memory Agent Model with CV support
//
// This model has additional structures to handle test data for cross-validation

functions{
  real normal_lb_rng(real mu, real sigma, real lb) {
    real p = normal_cdf(lb | mu, sigma);  // CDF for bounds
    real u = uniform_rng(p, 1);
    return (sigma * inv_Phi(u)) + mu;  // Inverse CDF for value
  }
}

// Input data - includes both training and test data
data {
 int<lower = 1> trials;              // Number of trials per agent
 int<lower = 1> agents;              // Number of training agents
 array[trials, agents] int h;        // Training choice data
 array[trials, agents] int other;    // Opponent's choices for training agents
 
 int<lower = 1> agents_test;         // Number of test agents
 array[trials, agents_test] int h_test;    // Test choice data
 array[trials, agents_test] int other_test; // Opponent's choices for test agents
}

// Parameters to be estimated
parameters {
  real biasM;                        // Population mean baseline bias
  real betaM;                        // Population mean memory sensitivity
  vector<lower = 0>[2] tau;          // Population SDs
  matrix[2, agents] z_IDs;           // Standardized individual parameters
  cholesky_factor_corr[2] L_u;       // Cholesky factor of correlation matrix
}

// Transformed parameters
transformed parameters {
  // Memory states for training data
  array[trials, agents] real memory;
  
  // Memory states for test data
  array[trials, agents_test] real memory_test;
  
  // Individual parameters
  matrix[agents,2] IDs;
  IDs = (diag_pre_multiply(tau, L_u) * z_IDs)';
  
  // Calculate memory states for training data
  for (agent in 1:agents){
    for (trial in 1:trials){
      if (trial == 1) {
        memory[trial, agent] = 0.5;
      } 
      if (trial < trials){
        memory[trial + 1, agent] = memory[trial, agent] + 
                                  ((other[trial, agent] - memory[trial, agent]) / trial);
        if (memory[trial + 1, agent] == 0){memory[trial + 1, agent] = 0.01;}
        if (memory[trial + 1, agent] == 1){memory[trial + 1, agent] = 0.99;}
      }
    }
  }
  
  // Calculate memory states for test data
  for (agent in 1:agents_test){
    for (trial in 1:trials){
      if (trial == 1) {
        memory_test[trial, agent] = 0.5;
      } 
      if (trial < trials){
        memory_test[trial + 1, agent] = memory_test[trial, agent] + 
                                      ((other_test[trial, agent] - memory_test[trial, agent]) / trial);
        if (memory_test[trial + 1, agent] == 0){memory_test[trial + 1, agent] = 0.01;}
        if (memory_test[trial + 1, agent] == 1){memory_test[trial + 1, agent] = 0.99;}
      }
    }
  }
}

// Model definition - trained only on training data
model {
  // Population-level priors
  target += normal_lpdf(biasM | 0, 1);
  target += normal_lpdf(tau[1] | 0, .3) - normal_lccdf(0 | 0, .3);
  target += normal_lpdf(betaM | 0, .3);
  target += normal_lpdf(tau[2] | 0, .3) - normal_lccdf(0 | 0, .3);
  target += lkj_corr_cholesky_lpdf(L_u | 2);

  // Standardized individual parameters
  target += std_normal_lpdf(to_vector(z_IDs));
  
  // Likelihood for training data only
  for (agent in 1:agents){
    for (trial in 1:trials){
      target += bernoulli_logit_lpmf(h[trial, agent] | 
                biasM + IDs[agent, 1] + memory[trial, agent] * (betaM + IDs[agent, 2]));
    }
  }
}

// Calculate log-likelihood for both training and test data
generated quantities{
   // Prior samples
   real biasM_prior;
   real<lower=0> biasSD_prior;
   real betaM_prior;
   real<lower=0> betaSD_prior;
   
   real bias_prior;
   real beta_prior;
   
   // Posterior predictive samples
   array[agents] int<lower=0, upper = trials> prior_preds0;
   array[agents] int<lower=0, upper = trials> prior_preds1;
   array[agents] int<lower=0, upper = trials> prior_preds2;
   array[agents] int<lower=0, upper = trials> posterior_preds0;
   array[agents] int<lower=0, upper = trials> posterior_preds1;
   array[agents] int<lower=0, upper = trials> posterior_preds2;
   
   // Log-likelihood for training data
   array[trials, agents] real log_lik;
   
   // Log-likelihood for test data - crucial for cross-validation
   array[trials, agents_test] real log_lik_test;
   
   // Generate prior samples
   biasM_prior = normal_rng(0,1);
   biasSD_prior = normal_lb_rng(0,0.3,0);
   betaM_prior = normal_rng(0,1);
   betaSD_prior = normal_lb_rng(0,0.3,0);
   
   bias_prior = normal_rng(biasM_prior, biasSD_prior);
   beta_prior = normal_rng(betaM_prior, betaSD_prior);
   
   // Generate predictions for different memory conditions
   for (i in 1:agents){
      prior_preds0[i] = binomial_rng(trials, inv_logit(bias_prior + 0 * beta_prior));
      prior_preds1[i] = binomial_rng(trials, inv_logit(bias_prior + 1 * beta_prior));
      prior_preds2[i] = binomial_rng(trials, inv_logit(bias_prior + 2 * beta_prior));
      posterior_preds0[i] = binomial_rng(trials, 
                           inv_logit(biasM + IDs[i,1] + 0 * (betaM + IDs[i,2])));
      posterior_preds1[i] = binomial_rng(trials, 
                           inv_logit(biasM + IDs[i,1] + 1 * (betaM + IDs[i,2])));
      posterior_preds2[i] = binomial_rng(trials, 
                           inv_logit(biasM + IDs[i,1] + 2 * (betaM + IDs[i,2])));
      
      // Calculate log-likelihood for training data
      for (t in 1:trials){
        log_lik[t,i] = bernoulli_logit_lpmf(h[t, i] | 
                      biasM + IDs[i, 1] + memory[t, i] * (betaM + IDs[i, 2]));
      }
   }
   
   // Calculate log-likelihood for test data
   // Note: We use population-level estimates for prediction
   for (i in 1:agents_test){
    for (t in 1:trials){
      log_lik_test[t,i] = bernoulli_logit_lpmf(h_test[t,i] | 
                         biasM + memory_test[t, i] * betaM);
    }
  }
}

