
// Multilevel Weighted Beta-Binomial Model
// This model allows different weights for different evidence sources
// Using total_weight and weight_prop parameterization

data {
  int<lower=1> N;                        // Total number of observations
  int<lower=1> J;                        // Number of subjects
  array[N] int<lower=1, upper=J> agent_id;  // Agent ID for each observation
  array[N] int<lower=0, upper=1> choice; // Choices (0=red, 1=blue)
  array[N] int<lower=0> blue1;           // Direct evidence (blue marbles)
  array[N] int<lower=0> total1;          // Total direct evidence
  array[N] int<lower=0> blue2;           // Social evidence (blue signals)
  array[N] int<lower=0> total2;          // Total social evidence
}

parameters {
  
  // Population-level parameters for agents' preconceptions
  real mu_alpha_prior;                   // Population mean for alpha prior
  real<lower=0> sigma_alpha_prior;       // Population SD for alpha prior
  real mu_beta_prior;                    // Population mean for beta prior
  real<lower=0> sigma_beta_prior;        // Population SD for beta prior
  
  // Population-level parameters
  real mu_weight_ratio;                  // Population mean for relative weight (direct/social) - log scale
  real mu_scaling;                       // Population mean for overall scaling - log scale
  
  // Population-level standard deviations
  real<lower=0> sigma_weight_ratio;      // Between-subject variability in relative weighting
  real<lower=0> sigma_scaling;           // Between-subject variability in scaling
  
  // Individual-level (random) effects
  vector[J] z_weight_ratio;              // Standardized individual weight ratio deviations
  vector[J] z_scaling;                   // Standardized individual scaling deviations
}

transformed parameters {
  // Individual-level parameters
  vector<lower=0>[J] weight_ratio;       // Individual relative weights (direct/social)
  vector<lower=0>[J] scaling_factor;     // Individual overall scaling factors
  vector<lower=0>[J] weight_direct;      // Individual weights for direct evidence
  vector<lower=0>[J] weight_social;      // Individual weights for social evidence
  
  // Non-centered parameterization
  for (j in 1:J) {
    // Transform standardized parameters to natural scale
    weight_ratio[j] = exp(mu_weight_ratio + z_weight_ratio[j] * sigma_weight_ratio);
    scaling_factor[j] = exp(mu_scaling + z_scaling[j] * sigma_scaling);
    
    // Calculate individual weights
    // The sum of weights is determined by the scaling factor
    // The ratio between weights is determined by weight_ratio
    weight_direct[j] = scaling_factor[j] * weight_ratio[j] / (1 + weight_ratio[j]);
    weight_social[j] = scaling_factor[j] / (1 + weight_ratio[j]);
  }
}

model {
  // Priors for population parameters
  mu_weight_ratio ~ normal(0, 1);        // Prior for log weight ratio centered at 0 (equal weights)
  mu_scaling ~ normal(0, 1);             // Prior for log scaling factor
  
  sigma_weight_ratio ~ exponential(2);   // Prior for between-subject variability
  sigma_scaling ~ exponential(2);        // Prior for scaling variability
  
  // Priors for individual random effects
  z_weight_ratio ~ std_normal();         // Standard normal prior for weight ratio z-scores
  z_scaling ~ std_normal();              // Standard normal prior for scaling z-scores
  z_alpha_prior ~ std_normal();          // Standard normal prior
  z_beta_prior ~ std_normal();           // Standard normal prior
  
  // Likelihood
  for (i in 1:N) {
    // Get weights for this person
    real w_direct = weight_direct[agent_id[i]];
    real w_social = weight_social[agent_id[i]];
    
    // Calculate weighted evidence
    real weighted_blue1 = blue1[i] * w_direct;
    real weighted_red1 = (total1[i] - blue1[i]) * w_direct;
    real weighted_blue2 = blue2[i] * w_social;
    real weighted_red2 = (total2[i] - blue2[i]) * w_social;
    
    // Calculate Beta parameters for Bayesian integration
    real alpha_post = alpha_prior[agent_id[i]] + weighted_blue1 + weighted_blue2;
    real beta_post = beta_prior[agent_id[i]] + weighted_red1 + weighted_red2;
    
    // Use beta-binomial distribution to model the choice
    target += beta_binomial_lpmf(choice[i] | 1, alpha_post, beta_post);
  }
}

generated quantities {
  // Convert population parameters to original weight scale for interpretation
  real population_ratio = exp(mu_weight_ratio);
  real population_scaling = exp(mu_scaling);
  real population_weight_direct = population_scaling * population_ratio / (1 + population_ratio);
  real population_weight_social = population_scaling / (1 + population_ratio);
  
  // Log likelihood for model comparison
  vector[N] log_lik;
  
  // Population and individual predictions
  array[N] int pred_choice;
  
  for (i in 1:N) {
    // Get weights for this person
    real w_direct = weight_direct[agent_id[i]];
    real w_social = weight_social[agent_id[i]];
    
    // Calculate weighted evidence
    real weighted_blue1 = blue1[i] * w_direct;
    real weighted_red1 = (total1[i] - blue1[i]) * w_direct;
    real weighted_blue2 = blue2[i] * w_social;
    real weighted_red2 = (total2[i] - blue2[i]) * w_social;
    
    // Calculate Beta parameters
    real alpha_post = alpha_prior[agent_id[i]] + weighted_blue1 + weighted_blue2;
    real beta_post = beta_prior[agent_id[i]] + weighted_red1 + weighted_red2;
    
    // Generate predictions using beta-binomial
    pred_choice[i] = beta_binomial_rng(1, alpha_post, beta_post);
    
    // Calculate log likelihood
    log_lik[i] = beta_binomial_lpmf(choice[i] | 1, alpha_post, beta_post);
  }
}

