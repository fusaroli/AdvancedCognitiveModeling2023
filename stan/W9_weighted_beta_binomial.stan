
data {
  int<lower=1> N;                        // Number of decisions
  array[N] int<lower=0, upper=1> choice; // Choices (0=red, 1=blue)
  array[N] int<lower=0> blue1;           // Direct evidence (blue marbles)
  array[N] int<lower=0> total1;          // Total direct evidence
  array[N] int<lower=0> blue2;           // Social evidence (blue signals)
  array[N] int<lower=0> total2;          // Total social evidence
}

parameters {
  // Using inverse logit for better numeric properties
  real<lower=0> weight_direct;           // Direct evidence weight
  real<lower=0> weight_social;           // Social evidence weight
}

model {
  // Priors
  target += normal_lpdf(weight_direct | 1, 0.3);  // Exponential prior with mean 1
  target += normal_lpdf(weight_social | 1, 0.3);  // Exponential prior with mean 1
  
  // Each observation is a separate decision
  for (i in 1:N) {
    // For this specific decision:
    real weighted_blue1 = blue1[i] * weight_direct;
    real weighted_red1 = (total1[i] - blue1[i]) * weight_direct;
    real weighted_blue2 = blue2[i] * weight_social;
    real weighted_red2 = (total2[i] - blue2[i]) * weight_social;
    
    // Calculate Beta parameters for this decision only
    real alpha_post = 1 + weighted_blue1 + weighted_blue2;
    real beta_post = 1 + weighted_red1 + weighted_red2;
    
    // Expected probability for this specific decision
    real expected_prob = alpha_post / (alpha_post + beta_post);
    
    // Model choice for this decision
    target += bernoulli_lpmf(choice[i] | expected_prob);
  }
}

generated quantities {
  // Log likelihood and predictions
  vector[N] log_lik;
  array[N] int posterior_pred_choice;
  
  for (i in 1:N) {
    real weighted_blue1 = blue1[i] * weight_direct;
    real weighted_red1 = (total1[i] - blue1[i]) * weight_direct;
    real weighted_blue2 = blue2[i] * weight_social;
    real weighted_red2 = (total2[i] - blue2[i]) * weight_social;
    
    real alpha_post = 1 + weighted_blue1 + weighted_blue2;
    real beta_post = 1 + weighted_red1 + weighted_red2;
    real expected_prob = alpha_post / (alpha_post + beta_post);
    
    log_lik[i] = bernoulli_lpmf(choice[i] | expected_prob);
    posterior_pred_choice[i] = bernoulli_rng(expected_prob);
  }
}

