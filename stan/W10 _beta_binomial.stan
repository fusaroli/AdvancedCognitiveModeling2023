
// Bayesian integration model relying on a beta-binomial distribution
// to preserve all uncertainty
// All evidence is taken at face value (equal weights)
data {
  int<lower=1> N;                      // Number of decisions
  array[N] int<lower=0, upper=1> choice; // Choices (0=red, 1=blue)
  array[N] int<lower=0> blue1;         // Direct evidence (blue marbles)
  array[N] int<lower=0> total1;        // Total direct evidence (total marbles)
  array[N] int<lower=0> blue2;         // Social evidence (blue signals)
  array[N] int<lower=0> total2;        // Total social evidence (total signals)
}

model {
  // Each observation is a separate decision
  for (i in 1:N) {
    // Calculate Beta parameters for posterior belief distribution
    real alpha_post = 1 + blue1[i] + blue2[i];
    real beta_post = 1 + (total1[i] - blue1[i]) + (total2[i] - blue2[i]);
    
    // Use beta_binomial distribution which integrates over all possible values
    // of the rate parameter weighted by their posterior probability
    target += beta_binomial_lpmf(choice[i] | 1, alpha_post, beta_post);
  }
}

generated quantities {
  // Log likelihood for model comparison
  vector[N] log_lik;
  
  // Prior and posterior predictive checks
  array[N] int prior_pred_choice;
  array[N] int posterior_pred_choice;
  
  for (i in 1:N) {
    // For prior predictions, use uniform prior (Beta(1,1))
    prior_pred_choice[i] = beta_binomial_rng(1, 1, 1);
    
    // For posterior predictions, use integrated evidence
    real alpha_post = 1 + blue1[i] + blue2[i];
    real beta_post = 1 + (total1[i] - blue1[i]) + (total2[i] - blue2[i]);
    
    // Generate predictions using the complete beta-binomial model
    posterior_pred_choice[i] = beta_binomial_rng(1, alpha_post, beta_post);
    
    // Log likelihood calculation using beta-binomial
    log_lik[i] = beta_binomial_lpmf(choice[i] | 1, alpha_post, beta_post);
  }
}

