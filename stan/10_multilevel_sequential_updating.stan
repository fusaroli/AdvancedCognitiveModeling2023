
// Multilevel Sequential Bayesian Updating Model
// This model captures individual differences in sequential belief updating
data {
  int<lower=1> N;                        // Total number of observations
  int<lower=1> J;                        // Number of agents
  int<lower=1> T;                        // Maximum number of trials per agent
  array[N] int<lower=1, upper=J> agent_id; // Agent ID for each observation
  array[N] int<lower=1, upper=T> trial_id; // Trial number for each observation
  array[N] int<lower=0, upper=1> choice;   // Choices (0=red, 1=blue)
  array[N] int<lower=0> blue1;             // Direct evidence (blue marbles)
  array[N] int<lower=0> total1;            // Total direct evidence
  array[N] int<lower=0> blue2;             // Social evidence (blue signals)
  array[N] int<lower=0> total2;            // Total social evidence
  // Additional data for tracking trial sequences
  array[J] int<lower=1, upper=T> trials_per_agent; // Number of trials for each agent
}

parameters {
  // Population-level parameters
  real mu_total_weight;                   // Population mean log total weight
  real mu_weight_prop_logit;              // Population mean logit weight proportion
  real mu_alpha_log;                      // Population mean log learning rate
  
  // Population-level standard deviations
  vector<lower=0>[3] tau;                 // SDs for [total_weight, weight_prop, alpha]
  
  // Correlation matrix for individual parameters (optional)
  cholesky_factor_corr[3] L_Omega;        // Cholesky factor of correlation matrix
  
  // Individual-level variations (non-centered parameterization)
  matrix[3, J] z;                         // Standardized individual parameters
}

transformed parameters {
  // Individual-level parameters
  vector<lower=0>[J] total_weight;        // Total evidence weight for each agent
  vector<lower=0, upper=1>[J] weight_prop; // Weight proportion for each agent
  vector<lower=0>[J] alpha;               // Learning rate for each agent
  vector<lower=0>[J] weight_direct;       // Direct evidence weight for each agent
  vector<lower=0>[J] weight_social;       // Social evidence weight for each agent
  
  // Individual beliefs for each trial
  // We'll use a ragged structure due to varying trial counts
  array[J, T] real belief;                // Belief in blue for each agent on each trial
  
  // Transform parameters to natural scale
  matrix[3, J] theta = diag_pre_multiply(tau, L_Omega) * z;  // Non-centered parameterization
  
  for (j in 1:J) {
    // Transform individual parameters to appropriate scales
    total_weight[j] = exp(mu_total_weight + theta[1, j]);
    weight_prop[j] = inv_logit(mu_weight_prop_logit + theta[2, j]);
    alpha[j] = exp(mu_alpha_log + theta[3, j]);
    
    // Calculate derived weights
    weight_direct[j] = total_weight[j] * weight_prop[j];
    weight_social[j] = total_weight[j] * (1 - weight_prop[j]);
    
    // Initialize belief tracking for each agent
    real alpha_param = 1.0;  // Initial beta distribution parameters
    real beta_param = 1.0;
    
    // Calculate initial belief
    belief[j, 1] = alpha_param / (alpha_param + beta_param);
    
    // Process trials for this agent (skipping the first trial since we initialized it above)
    for (t in 2:trials_per_agent[j]) {
      // Find the previous trial's data for this agent
      int prev_idx = 0;
      
      // Search for previous trial (this is a simplification; more efficient approaches exist)
      for (i in 1:N) {
        if (agent_id[i] == j && trial_id[i] == t-1) {
          prev_idx = i;
          break;
        }
      }
      
      if (prev_idx > 0) {
        // Calculate weighted evidence from previous trial
        real weighted_blue1 = blue1[prev_idx] * weight_direct[j];
        real weighted_red1 = (total1[prev_idx] - blue1[prev_idx]) * weight_direct[j];
        real weighted_blue2 = blue2[prev_idx] * weight_social[j];
        real weighted_red2 = (total2[prev_idx] - blue2[prev_idx]) * weight_social[j];
        
        // Update belief parameters with learning rate
        alpha_param = alpha_param + alpha[j] * (weighted_blue1 + weighted_blue2);
        beta_param = beta_param + alpha[j] * (weighted_red1 + weighted_red2);
      }
      
      // Calculate updated belief
      belief[j, t] = alpha_param / (alpha_param + beta_param);
    }
  }
}

model {
  // Priors for population parameters
  target += normal_lpdf(mu_total_weight | 0, 1);         // Population log total weight
  target += normal_lpdf(mu_weight_prop_logit | 0, 1);    // Population logit weight proportion
  target += normal_lpdf(mu_alpha_log | -1, 1);           // Population log learning rate
  
  // Priors for population standard deviations
  target += exponential_lpdf(tau | 2);                   // Conservative prior for SDs
  
  // Prior for correlation matrix
  target += lkj_corr_cholesky_lpdf(L_Omega | 2);         // LKJ prior on correlations
  
  // Prior for standardized individual parameters
  target += std_normal_lpdf(to_vector(z));               // Standard normal prior on z-scores
  
  // Likelihood
  for (i in 1:N) {
    int j = agent_id[i];                                  // Agent ID
    int t = trial_id[i];                                  // Trial number
    
    // Model choice as a function of current belief
    target += bernoulli_lpmf(choice[i] | belief[j, t]);
  }
}

generated quantities {
  // Transform population parameters to natural scale for interpretation
  real<lower=0> pop_total_weight = exp(mu_total_weight);
  real<lower=0, upper=1> pop_weight_prop = inv_logit(mu_weight_prop_logit);
  real<lower=0> pop_alpha = exp(mu_alpha_log);
  real<lower=0> pop_weight_direct = pop_total_weight * pop_weight_prop;
  real<lower=0> pop_weight_social = pop_total_weight * (1 - pop_weight_prop);
  
  // Correlation matrix for individual differences
  matrix[3, 3] Omega = multiply_lower_tri_self_transpose(L_Omega);
  
  // Log likelihood for model comparison
  vector[N] log_lik;
  
  // Posterior predictions
  array[N] int pred_choice;
  
  for (i in 1:N) {
    int j = agent_id[i];
    int t = trial_id[i];
    
    // Generate predicted choices
    pred_choice[i] = bernoulli_rng(belief[j, t]);
    
    // Calculate log likelihood
    log_lik[i] = bernoulli_lpmf(choice[i] | belief[j, t]);
  }
}

