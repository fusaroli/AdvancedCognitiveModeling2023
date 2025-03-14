
// Bayesian integration model with separate theta parameters for each evidence source
// All evidence is taken at face value (equal weights)
data {
  int<lower=1> N;                        // Number of decisions
  array[N] int<lower=0, upper=1> choice; // Choices (0=red, 1=blue)
  array[N] int<lower=0> blue1;           // Direct evidence (blue marbles)
  array[N] int<lower=0> total1;          // Total direct evidence (total marbles)
  array[N] int<lower=0> blue2;           // Social evidence (blue signals)
  array[N] int<lower=0> total2;          // Total social evidence (total signals)
}

model {
  // Each observation is a separate decision
  for (i in 1:N) {
    // Calculate Beta parameters for this decision only (including a flat prior)
    real alpha_post = 1 + blue1[i] + blue2[i];
    real beta_post = 1 + (total1[i] - blue1[i]) + (total2[i] - blue2[i]);
    
    // Expected probability for this specific decision
    real expected_prob = alpha_post / (alpha_post + beta_post);
    
    // Model choice for this decision
    target += bernoulli_lpmf(choice[i] | expected_prob);
  }
}

generated quantities {
  // Log likelihood for model comparison
  vector[N] log_lik;
  
  // Prior samples for model checking
  array[N] int prior_pred_choice;
  
  // Predictive checks
  array[N] int posterior_pred_choice;
  
  for (i in 1:N) {
    // For prior predictions, use uniform prior (Beta(1,1))
    real prior_prob = 0.5;  // Expected value of Beta(1,1)
    
    // For posterior predictions, use integrated evidence
    real alpha_post = 1 + blue1[i] + blue2[i];
    real beta_post = 1 + (total1[i] - blue1[i]) + (total2[i] - blue2[i]);
    real expected_prob = alpha_post / (alpha_post + beta_post);
    
    // Generate predictions
    prior_pred_choice[i] = bernoulli_rng(prior_prob);
    posterior_pred_choice[i] = bernoulli_rng(expected_prob);
    
    // Log likelihood calculation
    log_lik[i] = bernoulli_lpmf(choice[i] | expected_prob);
  }
}

