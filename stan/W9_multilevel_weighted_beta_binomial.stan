
// Multilevel Weighted Beta-Binomial Model with Relative Weighting
// This model parameterizes weights in relation to each other for better identifiability

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
  // Population-level (fixed) effects
  real mu_weight_ratio;                  // Population mean for relative weight (direct/social)
  
  // Population-level standard deviation
  real<lower=0> sigma_weight_ratio;      // Between-subject variability in relative weighting
  
  // Individual-level (random) effects
  vector[J] z;                           // Standardized individual deviations
  
  // Overall evidence scaling factor (shared across both sources)
  real<lower=0> mu_scaling;              // Population mean for overall scaling
  real<lower=0> sigma_scaling;           // Between-subject variability in scaling
  vector[J] z_scaling;                   // Individual scaling factors (standardized)
}

transformed parameters {
  // Individual-level parameters
  vector<lower=0>[J] weight_ratio;       // Individual relative weights (direct/social)
  vector<lower=0>[J] scaling_factor;     // Individual overall scaling factors
  vector<lower=0>[J] weight_direct;      // Individual weights for direct evidence
  vector<lower=0>[J] weight_social;      // Individual weights for social evidence
  
  // Non-centered parameterization for weight ratio
  for (j in 1:J) {
    // Transform standardized parameters to natural scale
    weight_ratio[j] = exp(mu_weight_ratio + z[j] * sigma_weight_ratio);
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
  sigma_weight_ratio ~ exponential(2);   // Prior for between-subject variability
  
  mu_scaling ~ normal(0, 1);             // Prior for log scaling factor
  sigma_scaling ~ exponential(2);        // Prior for scaling variability
  
  // Priors for individual random effects
  z ~ std_normal();                      // Standard normal prior for weight ratio z-scores
  z_scaling ~ std_normal();              // Standard normal prior for scaling z-scores
  
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
    real alpha_post = 1 + weighted_blue1 + weighted_blue2;
    real beta_post = 1 + weighted_red1 + weighted_red2;
    
    // Expected probability from integrated evidence
    real expected_prob = alpha_post / (alpha_post + beta_post);
    
    // Model choice
    choice[i] ~ bernoulli(expected_prob);
  }
}

generated quantities {
  // Log likelihood for model comparison
  vector[N] log_lik;
  
  // Population and individual predictions
  array[N] int pred_choice;
  
  // Convert population parameters to original weight scale for interpretation
  real population_ratio = exp(mu_weight_ratio);
  real population_scaling = exp(mu_scaling);
  real population_weight_direct = population_scaling * population_ratio / (1 + population_ratio);
  real population_weight_social = population_scaling / (1 + population_ratio);
  
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
    real alpha_post = 1 + weighted_blue1 + weighted_blue2;
    real beta_post = 1 + weighted_red1 + weighted_red2;
    
    // Expected probability
    real expected_prob = alpha_post / (alpha_post + beta_post);
    
    // Generate predictions and calculate log likelihood
    pred_choice[i] = bernoulli_rng(expected_prob);
    log_lik[i] = bernoulli_lpmf(choice[i] | expected_prob);
  }
}

