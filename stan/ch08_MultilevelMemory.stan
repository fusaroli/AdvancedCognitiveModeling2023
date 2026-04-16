
// Multilevel Memory Agent Model
//
// This model assumes agents make choices based on their memory of
// the opponent's previous choices.

functions{
  // Helper function for generating truncated normal random numbers
  real normal_lb_rng(real mu, real sigma, real lb) {
    real p = normal_cdf(lb | mu, sigma);  // CDF for bounds
    real u = uniform_rng(p, 1);
    return (sigma * inv_Phi(u)) + mu;  // Inverse CDF for value
  }
}

// Input data
data {
 int<lower = 1> trials;         // Number of trials per agent
 int<lower = 1> agents;         // Number of agents
 array[trials, agents] int h;   // Choice data (0/1 for each trial and agent)
 array[trials, agents] int other;// Opponent's choices (input to memory)
}

// Parameters to be estimated
parameters {
  real biasM;                   // Population mean baseline bias
  real betaM;                   // Population mean memory sensitivity
  vector<lower = 0>[2] tau;     // Population SDs for bias and beta
  matrix[2, agents] z_IDs;      // Standardized individual parameters (non-centered)
  cholesky_factor_corr[2] L_u;  // Cholesky factor of correlation matrix
}

// Transformed parameters (derived quantities)
transformed parameters {
  // Memory state for each agent and trial
  array[trials, agents] real memory;
  
  // Individual parameters (bias and beta for each agent)
  matrix[agents, 2] IDs;
  
  // Transform standardized parameters to actual parameters (non-centered parameterization)
  IDs = (diag_pre_multiply(tau, L_u) * z_IDs)';
  
  // Calculate memory states based on opponent's choices
  for (agent in 1:agents){
    for (trial in 1:trials){
      // Initialize first trial with neutral memory
      if (trial == 1) {
        memory[trial, agent] = 0.5;
      } 
      // Update memory based on opponent's choices
      if (trial < trials){
        // Simple averaging memory update
        memory[trial + 1, agent] = memory[trial, agent] + 
                                ((other[trial, agent] - memory[trial, agent]) / trial);
        
        // Handle edge cases to avoid numerical issues
        if (memory[trial + 1, agent] == 0){memory[trial + 1, agent] = 0.01;}
        if (memory[trial + 1, agent] == 1){memory[trial + 1, agent] = 0.99;}
      }
    }
  }
}

// Model definition
model {
  // Population-level priors
  target += normal_lpdf(biasM | 0, 1);
  target += normal_lpdf(tau[1] | 0, .3) - normal_lccdf(0 | 0, .3);  // Half-normal for SD
  target += normal_lpdf(betaM | 0, .3);
  target += normal_lpdf(tau[2] | 0, .3) - normal_lccdf(0 | 0, .3);  // Half-normal for SD
  
  // Prior for correlation matrix
  target += lkj_corr_cholesky_lpdf(L_u | 2);

  // Standardized individual parameters have standard normal prior
  target += std_normal_lpdf(to_vector(z_IDs));
  
  // Likelihood: predict each agent's choices
  for (agent in 1:agents){
    for (trial in 1:trials){
      // choice ~ bias + memory_effect*beta
      target += bernoulli_logit_lpmf(h[trial, agent] | 
                biasM + IDs[agent, 1] + memory[trial, agent] * (betaM + IDs[agent, 2]));
    }
  }
}

// Additional quantities to calculate
generated quantities{
   // Prior predictive samples
   real biasM_prior;
   real<lower=0> biasSD_prior;
   real betaM_prior;
   real<lower=0> betaSD_prior;
   
   real bias_prior;
   real beta_prior;
   
   // Posterior predictive samples for different memory conditions
   array[agents] int<lower=0, upper = trials> prior_preds0;
   array[agents] int<lower=0, upper = trials> prior_preds1;
   array[agents] int<lower=0, upper = trials> prior_preds2;
   array[agents] int<lower=0, upper = trials> posterior_preds0;
   array[agents] int<lower=0, upper = trials> posterior_preds1;
   array[agents] int<lower=0, upper = trials> posterior_preds2;
   
   // Log-likelihood for each observation (crucial for model comparison)
   array[trials, agents] real log_lik;
   
   // Generate prior samples
   biasM_prior = normal_rng(0,1);
   biasSD_prior = normal_lb_rng(0,0.3,0);
   betaM_prior = normal_rng(0,1);
   betaSD_prior = normal_lb_rng(0,0.3,0);
   
   bias_prior = normal_rng(biasM_prior, biasSD_prior);
   beta_prior = normal_rng(betaM_prior, betaSD_prior);
   
   // Generate predictions for different memory conditions
   for (i in 1:agents){
      // Prior predictions for low, medium, high memory
      prior_preds0[i] = binomial_rng(trials, inv_logit(bias_prior + 0 * beta_prior));
      prior_preds1[i] = binomial_rng(trials, inv_logit(bias_prior + 1 * beta_prior));
      prior_preds2[i] = binomial_rng(trials, inv_logit(bias_prior + 2 * beta_prior));
      
      // Posterior predictions for low, medium, high memory
      posterior_preds0[i] = binomial_rng(trials, 
                           inv_logit(biasM + IDs[i,1] + 0 * (betaM + IDs[i,2])));
      posterior_preds1[i] = binomial_rng(trials, 
                           inv_logit(biasM + IDs[i,1] + 1 * (betaM + IDs[i,2])));
      posterior_preds2[i] = binomial_rng(trials, 
                           inv_logit(biasM + IDs[i,1] + 2 * (betaM + IDs[i,2])));
      
      // Calculate log-likelihood for each observation
      for (t in 1:trials){
        log_lik[t,i] = bernoulli_logit_lpmf(h[t, i] | 
                      biasM + IDs[i, 1] + memory[t, i] * (betaM + IDs[i, 2]));
      }
   }
}

