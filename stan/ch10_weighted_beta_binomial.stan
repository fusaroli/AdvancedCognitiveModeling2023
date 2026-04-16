
data {
  int<lower=1> N;                        // Number of decisions
  array[N] int<lower=0, upper=1> choice; // Choices (0=red, 1=blue)
  array[N] int<lower=0> blue1;           // Direct evidence (blue marbles)
  array[N] int<lower=0> total1;          // Total direct evidence
  array[N] int<lower=0> blue2;           // Social evidence (blue signals)
  array[N] int<lower=0> total2;          // Total social evidence
}

parameters {
  real<lower=0> total_weight;         // Total influence of all evidence
  real<lower=0, upper=1> weight_prop; // Proportion of weight for direct evidence
}

transformed parameters {
  real<lower=0> weight_direct = total_weight * weight_prop;
  real<lower=0> weight_social = total_weight * (1 - weight_prop);
}

model {
  // Priors
  target += gamma_lpdf(total_weight | 2, 1);  // Centered around 2 with reasonable spread
  target += beta_lpdf(weight_prop | 1, 1);    // Uniform prior on proportion
  
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
  real weight_direct_prior = normal_rng(1, 0.3);
  real weight_social_prior = normal_rng(1, 0.3);
  
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

