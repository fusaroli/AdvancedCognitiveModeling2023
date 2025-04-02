
data {
  int<lower=1> N;                        // Number of decisions
  array[N] int<lower=0, upper=1> choice; // Choices (0=red, 1=blue)
  array[N] int<lower=0> blue1;           // Direct evidence (blue marbles)
  array[N] int<lower=0> total1;          // Total direct evidence
  array[N] int<lower=0> blue2;           // Social evidence (blue signals)
  array[N] int<lower=0> total2;          // Total social evidence
}

parameters {
  real<lower = 0> alpha_prior;                    // Prior alpha parameter
  real<lower = 0> beta_prior;                     // Prior beta parameter
  real<lower=0> total_weight;         // Total influence of all evidence
  real<lower=0, upper=1> weight_prop; // Proportion of weight for direct evidence
}

transformed parameters {
  real<lower=0> weight_direct = total_weight * weight_prop;
  real<lower=0> weight_social = total_weight * (1 - weight_prop);
}

model {
  // Priors
  target += lognormal_lpdf(alpha_prior | 0, 1); // Prior on alpha_prior
  target += lognormal_lpdf(beta_prior | 0, 1);  // Prior on beta_prior
  target += lognormal_lpdf(total_weight | .8, .4);  // Centered around 2 with reasonable spread and always positive
  target += beta_lpdf(weight_prop | 1, 1);    // Uniform prior on proportion
  
  // Each observation is a separate decision
  for (i in 1:N) {
    // For this specific decision:
    real weighted_blue1 = blue1[i] * weight_direct;
    real weighted_red1 = (total1[i] - blue1[i]) * weight_direct;
    real weighted_blue2 = blue2[i] * weight_social;
    real weighted_red2 = (total2[i] - blue2[i]) * weight_social;
    
    // Calculate Beta parameters for this decision
    real alpha_post = alpha_prior + weighted_blue1 + weighted_blue2;
    real beta_post = beta_prior + weighted_red1 + weighted_red2;
    
    // Use beta_binomial distribution to integrate over the full posterior
    target += beta_binomial_lpmf(choice[i] | 1, alpha_post, beta_post);
  }
}

generated quantities {
  // Log likelihood and predictions
  vector[N] log_lik;
  array[N] int posterior_pred_choice;
  array[N] int prior_pred_choice;
  
  // Sample the agent's preconceptions
  real alpha_prior_prior = lognormal_rng(0, 1);
  real beta_prior_prior = lognormal_rng(0, 1);
  
  // Sample from priors for the reparameterized model
  real<lower = 0> total_weight_prior = lognormal_rng(.8, .4);
  real weight_prop_prior = beta_rng(1, 1);
  
  // Derive the implied direct and social weights from the prior samples
  real weight_direct_prior = total_weight_prior * weight_prop_prior;
  real weight_social_prior = total_weight_prior * (1 - weight_prop_prior);
  
  // Posterior predictions and log-likelihood
  for (i in 1:N) {
    // Posterior predictions using the weighted evidence
    real weighted_blue1 = blue1[i] * weight_direct;
    real weighted_red1 = (total1[i] - blue1[i]) * weight_direct;
    real weighted_blue2 = blue2[i] * weight_social;
    real weighted_red2 = (total2[i] - blue2[i]) * weight_social;
    
    real alpha_post = alpha_prior + weighted_blue1 + weighted_blue2;
    real beta_post = beta_prior + weighted_red1 + weighted_red2;
    
    // Log likelihood using beta_binomial
    log_lik[i] = beta_binomial_lpmf(choice[i] | 1, alpha_post, beta_post);
    
    // Generate predictions from the full posterior
    posterior_pred_choice[i] = beta_binomial_rng(1, alpha_post, beta_post);
    
    // Prior predictions using the prior-derived weights
    real prior_weighted_blue1 = blue1[i] * weight_direct_prior;
    real prior_weighted_red1 = (total1[i] - blue1[i]) * weight_direct_prior;
    real prior_weighted_blue2 = blue2[i] * weight_social_prior;
    real prior_weighted_red2 = (total2[i] - blue2[i]) * weight_social_prior;
    
    real alpha_prior_preds = alpha_prior + prior_weighted_blue1 + prior_weighted_blue2;
    real beta_prior_preds = beta_prior + prior_weighted_red1 + prior_weighted_red2;
    
    // Generate predictions from the prior
    prior_pred_choice[i] = beta_binomial_rng(1, alpha_prior, beta_prior);
  }
}

