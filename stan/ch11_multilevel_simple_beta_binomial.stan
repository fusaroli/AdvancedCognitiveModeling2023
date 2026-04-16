
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
  // Population-level parameters for agents' preconceptions
  real mu_alpha_prior;                   // Population mean for alpha prior
  real<lower=0> sigma_alpha_prior;       // Population SD for alpha prior
  real mu_beta_prior;                    // Population mean for beta prior
  real<lower=0> sigma_beta_prior;        // Population SD for beta prior
  
  // Population-level parameter for overall scaling
  real mu_scaling;                       // Population mean scaling factor (log scale)
  real<lower=0> sigma_scaling;           // Population SD of scaling
  
  // Individual-level (random) effects
  vector[J] z_alpha_prior;               // Standardized individual deviations for alpha prior
  vector[J] z_beta_prior;                // Standardized individual deviations for beta prior
  vector[J] z_scaling;                   // Standardized individual deviations
  
}

transformed parameters {
  // Individual-level parameters
  vector<lower=0>[J] scaling_factor;     // Individual scaling factors
  vector<lower=0>[J] alpha_prior;        // Individual alpha prior
  vector<lower=0>[J] beta_prior;         // Individual beta prior
  
  // Non-centered parameterization for scaling factor
  for (j in 1:J) {
    alpha_prior[j] = exp(mu_alpha_prior + z_alpha_prior[j] * sigma_alpha_prior);
    beta_prior[j] = exp(mu_beta_prior + z_beta_prior[j] * sigma_beta_prior);
    scaling_factor[j] = exp(mu_scaling + z_scaling[j] * sigma_scaling);
  }
}

model {
  // Priors for population parameters
  target += lognormal_lpdf(mu_alpha_prior | 0, 1);         // Prior for population mean of alpha prior
  target += exponential_lpdf(sigma_alpha_prior | 1);       // Prior for population SD of alpha prior
  target += lognormal_lpdf(mu_beta_prior | 0, 1);          // Prior for population mean of beta prior
  target += exponential_lpdf(sigma_beta_prior | 1);        // Prior for population SD of beta prior
  target += normal_lpdf(mu_scaling | 0, 1);             // Prior for log scaling factor
  target += exponential_lpdf(sigma_scaling | 2);        // Prior for between-subject variability
  
  // Prior for standardized random effects
  z_scaling ~ std_normal();              // Standard normal prior
  z_alpha_prior ~ std_normal();          // Standard normal prior
  z_beta_prior ~ std_normal();           // Standard normal prior
  
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
    real alpha_post = alpha_prior[agent_id[i]] + weighted_blue1 + weighted_blue2;
    real beta_post = beta_prior[agent_id[i]] + weighted_red1 + weighted_red2;
    
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
    real alpha_post = alpha_prior[agent_id[i]] + weighted_blue1 + weighted_blue2;
    real beta_post = beta_prior[agent_id[i]] + weighted_red1 + weighted_red2;
    
    // Generate predictions using beta-binomial
    pred_choice[i] = beta_binomial_rng(1, alpha_post, beta_post);
    
    // Calculate log likelihood
    log_lik[i] = beta_binomial_lpmf(choice[i] | 1, alpha_post, beta_post);
  }
}

