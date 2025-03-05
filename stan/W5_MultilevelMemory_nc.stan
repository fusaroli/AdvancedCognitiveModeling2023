
// Multilevel Memory Agent Model (Non-Centered Parameterization)
//
functions{
  real normal_lb_rng(real mu, real sigma, real lb) {
    real p = normal_cdf(lb | mu, sigma);  // cdf for bounds
    real u = uniform_rng(p, 1);
    return (sigma * inv_Phi(u)) + mu;  // inverse cdf for value
  }
}

// The input data for the model
data {
  int<lower = 1> trials;  // Number of trials per agent
  int<lower = 1> agents;  // Number of agents
  array[trials, agents] int h;  // Memory agent choices
  array[trials, agents] int other;  // Opponent (random agent) choices
}

// Parameters to be estimated
parameters {
  // Population-level parameters
  real biasM;               // Mean of baseline bias
  real<lower = 0> biasSD;   // SD of baseline bias
  real betaM;               // Mean of memory sensitivity
  real<lower = 0> betaSD;   // SD of memory sensitivity
  
  // Standardized individual parameters (non-centered parameterization)
  vector[agents] biasID_z;  // Standardized individual bias parameters
  vector[agents] betaID_z;  // Standardized individual beta parameters
}

// Transformed parameters (derived quantities)
transformed parameters {
  // Memory state for each agent and trial
  array[trials, agents] real memory;
  
  // Individual parameters (constructed from non-centered parameterization)
  vector[agents] biasID;
  vector[agents] betaID;
  
  // Calculate memory states based on opponent's choices
  for (agent in 1:agents){
    for (trial in 1:trials){
      // Initial memory state (no prior information)
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
  
  // Construct individual parameters from non-centered parameterization
  biasID = biasM + biasID_z * biasSD;
  betaID = betaM + betaID_z * betaSD;
}

// Model definition
model {
  // Population-level priors
  target += normal_lpdf(biasM | 0, 1);
  target += normal_lpdf(biasSD | 0, .3) - normal_lccdf(0 | 0, .3);  // Half-normal
  target += normal_lpdf(betaM | 0, .3);
  target += normal_lpdf(betaSD | 0, .3) - normal_lccdf(0 | 0, .3);  // Half-normal

  // Standardized individual parameters (non-centered parameterization)
  target += std_normal_lpdf(biasID_z);  // Standard normal prior for z-scores
  target += std_normal_lpdf(betaID_z);  // Standard normal prior for z-scores
 
  // Likelihood
  for (agent in 1:agents){
    for (trial in 1:trials){
      target += bernoulli_logit_lpmf(h[trial,agent] | 
            biasID[agent] + logit(memory[trial, agent]) * betaID[agent]);
    }
  }
}

// Generated quantities for model checking and predictions
generated quantities{
   // Prior samples for checking
   real biasM_prior;
   real<lower=0> biasSD_prior;
   real betaM_prior;
   real<lower=0> betaSD_prior;
   
   real bias_prior;
   real beta_prior;
   
   // Predictive simulations with different memory values (individual level)
   array[agents] int<lower=0, upper = trials> prior_preds0;  // No memory effect (memory=0)
   array[agents] int<lower=0, upper = trials> prior_preds1;  // Neutral memory (memory=0.5)
   array[agents] int<lower=0, upper = trials> prior_preds2;  // Strong memory (memory=1)
   array[agents] int<lower=0, upper = trials> posterior_preds0;
   array[agents] int<lower=0, upper = trials> posterior_preds1;
   array[agents] int<lower=0, upper = trials> posterior_preds2;
   
   // Generate prior samples
   biasM_prior = normal_rng(0,1);
   biasSD_prior = normal_lb_rng(0,0.3,0);
   betaM_prior = normal_rng(0,1);
   betaSD_prior = normal_lb_rng(0,0.3,0);
   
   bias_prior = normal_rng(biasM_prior, biasSD_prior);
   beta_prior = normal_rng(betaM_prior, betaSD_prior);
   
   // Generate predictions for each agent
   for (agent in 1:agents){
      // Prior predictive checks
      prior_preds0[agent] = binomial_rng(trials, inv_logit(bias_prior + 0 * beta_prior));
      prior_preds1[agent] = binomial_rng(trials, inv_logit(bias_prior + 1 * beta_prior));
      prior_preds2[agent] = binomial_rng(trials, inv_logit(bias_prior + 2 * beta_prior));
      
      // Posterior predictive checks
      posterior_preds0[agent] = binomial_rng(trials, 
                              inv_logit(biasM + biasID[agent] + 0 * (betaM + betaID[agent])));
      posterior_preds1[agent] = binomial_rng(trials, 
                              inv_logit(biasM + biasID[agent] + 1 * (betaM + betaID[agent])));
      posterior_preds2[agent] = binomial_rng(trials, 
                              inv_logit(biasM + biasID[agent] + 2 * (betaM + betaID[agent])));
   }
}

