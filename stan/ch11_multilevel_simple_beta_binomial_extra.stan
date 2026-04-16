
// Multilevel Simple Beta-Binomial Model
// This model assumes equal weights for evidence sources (taking evidence at face value)
// but allows for individual variation in overall responsiveness

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
  // Population-level parameter for overall scaling
  real mu_scaling;                       // Population mean scaling factor (log scale)
  real<lower=0> sigma_scaling;           // Population SD of scaling
  
  // Individual-level (random) effects
  vector[J] z_scaling;                   // Standardized individual deviations
}

transformed parameters {
  // Individual-level parameters
  vector<lower=0>[J] scaling_factor;     // Individual scaling factors
  
  // Non-centered parameterization for scaling factor
  for (j in 1:J) {
    scaling_factor[j] = exp(mu_scaling + z_scaling[j] * sigma_scaling);
  }
}

model {
  // Priors for population parameters
  mu_scaling ~ normal(0, 1);             // Prior for log scaling factor
  sigma_scaling ~ exponential(2);        // Prior for between-subject variability
  
  // Prior for standardized random effects
  z_scaling ~ std_normal();              // Standard normal prior
  
  // Likelihood
  for (i in 1:N) {
    // Calculate the individual scaling factor
    real scale = scaling_factor[agent_id[i]];
    
    // Simple integration - weights both evidence sources equally but applies individual scaling
    // Both direct and social evidence get weight = 1.0 * scaling_factor
    real weighted_blue1 = blue1[i] * scale;
    real weighted_red1 = (total1[i] - blue1[i]) * scale;
    real weighted_blue2 = blue2[i] * scale;
    real weighted_red2 = (total2[i] - blue2[i]) * scale;
    
    // Calculate Beta parameters for posterior
    real alpha_post = 1 + weighted_blue1 + weighted_blue2;
    real beta_post = 1 + weighted_red1 + weighted_red2;
    
    // Use beta-binomial distribution to model the choice
    target += beta_binomial_lpmf(choice[i] | 1, alpha_post, beta_post);
  }
}

generated quantities {
  // Population parameters on natural scale
  real population_scaling = exp(mu_scaling);
  
  // Log likelihood for model comparison
  vector[N] log_lik;
  
  // Population and individual predictions
  array[N] int pred_choice;
  
  for (i in 1:N) {
    // Calculate the individual scaling factor
    real scale = scaling_factor[agent_id[i]];
    
    // Calculate weighted evidence
    real weighted_blue1 = blue1[i] * scale;
    real weighted_red1 = (total1[i] - blue1[i]) * scale;
    real weighted_blue2 = blue2[i] * scale;
    real weighted_red2 = (total2[i] - blue2[i]) * scale;
    
    // Calculate Beta parameters
    real alpha_post = 1 + weighted_blue1 + weighted_blue2;
    real beta_post = 1 + weighted_red1 + weighted_red2;
    
    // Generate predictions using beta-binomial
    pred_choice[i] = beta_binomial_rng(1, alpha_post, beta_post);
    
    // Calculate log likelihood
    log_lik[i] = beta_binomial_lpmf(choice[i] | 1, alpha_post, beta_post);
  }
}

