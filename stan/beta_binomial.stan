
data {
  int<lower=1> N;                 // Number of decisions
  array[N] int<lower=0, upper=1> choice;   // Choices (0=red, 1=blue)
  array[N] int<lower=0> blue1;    // Direct evidence (blue marbles)
  array[N] int<lower=0> total1;   // Total direct evidence (total marbles)
  array[N] int<lower=0> blue2;    // Social evidence (blue signals)
  array[N] int<lower=0> total2;   // Total social evidence (total signals)
}

parameters {
  real<lower=0, upper=10> alpha_prior;  // Prior alpha parameter
  real<lower=0, upper=10> beta_prior;   // Prior beta parameter
}

model {
  // Weakly informative priors
  alpha_prior ~ normal(1, 0.5);
  beta_prior ~ normal(1, 0.5);
  
  // Likelihood
  for (i in 1:N) {
    // Calculate posterior beta parameters
    real alpha_post = alpha_prior + blue1[i] + blue2[i];
    real beta_post = beta_prior + (total1[i] - blue1[i]) + (total2[i] - blue2[i]);
    
    // Calculate expected probability of blue
    real p_blue = alpha_post / (alpha_post + beta_post);
    
    // Likelihood of choice
    choice[i] ~ bernoulli(p_blue);
  }
}

generated quantities {
  // Log likelihood for model comparison
  vector[N] log_lik;
  
  // Prior samples for prior predictive checks
  real<lower=0, upper=10> alpha_prior_sample = normal_rng(1, 0.5);
  real<lower=0, upper=10> beta_prior_sample = normal_rng(1, 0.5);
  
  // Posterior predictions for each observation
  array[N] int posterior_pred;
  
  // Prior predictions for each observation
  array[N] int prior_pred;
  
  // Expected probabilities
  array[N] real expected_prob;
  array[N] real prior_prob;
  
  // Distribution parameters for selected examples
  // We'll track 3 representative cases from the dataset
  array[3] real alpha_post_examples;
  array[3] real beta_post_examples;
  array[3] real alpha_prior_examples;
  array[3] real beta_prior_examples;
  
  // Calculate all generated quantities
  for (i in 1:N) {
    // Calculate posterior beta parameters
    real alpha_post = alpha_prior + blue1[i] + blue2[i];
    real beta_post = beta_prior + (total1[i] - blue1[i]) + (total2[i] - blue2[i]);
    
    // Calculate posterior probability
    real p_blue = alpha_post / (alpha_post + beta_post);
    expected_prob[i] = p_blue;
    
    // Calculate prior probability 
    real alpha_post_prior = alpha_prior_sample + blue1[i] + blue2[i];
    real beta_post_prior = beta_prior_sample + (total1[i] - blue1[i]) + (total2[i] - blue2[i]);
    prior_prob[i] = alpha_post_prior / (alpha_post_prior + beta_post_prior);
    
    // Generate posterior prediction
    posterior_pred[i] = bernoulli_rng(p_blue);
    
    // Generate prior prediction
    prior_pred[i] = bernoulli_rng(prior_prob[i]);
    
    // Log likelihood for model comparison
    log_lik[i] = bernoulli_lpmf(choice[i] | p_blue);
  }
  
  // Store distribution parameters for 3 representative cases 
  // (we'll pick cases with different evidence combinations)
  // Find some interesting indices (this is simplified - in practice, choose meaningful cases)
  {
    // Choose specific indices based on evidence patterns
    int idx1 = 1;  // For example, a case with low direct & low social evidence
    int idx2 = 40; // For example, a case with mixed evidence 
    int idx3 = 80; // For example, a case with high direct & high social evidence
    
    // Store posterior parameters
    alpha_post_examples[1] = alpha_prior + blue1[idx1] + blue2[idx1];
    beta_post_examples[1] = beta_prior + (total1[idx1] - blue1[idx1]) + (total2[idx1] - blue2[idx1]);
    
    alpha_post_examples[2] = alpha_prior + blue1[idx2] + blue2[idx2];
    beta_post_examples[2] = beta_prior + (total1[idx2] - blue1[idx2]) + (total2[idx2] - blue2[idx2]);
    
    alpha_post_examples[3] = alpha_prior + blue1[idx3] + blue2[idx3];
    beta_post_examples[3] = beta_prior + (total1[idx3] - blue1[idx3]) + (total2[idx3] - blue2[idx3]);
    
    // Store prior parameters
    alpha_prior_examples[1] = alpha_prior_sample + blue1[idx1] + blue2[idx1];
    beta_prior_examples[1] = beta_prior_sample + (total1[idx1] - blue1[idx1]) + (total2[idx1] - blue2[idx1]);
    
    alpha_prior_examples[2] = alpha_prior_sample + blue1[idx2] + blue2[idx2];
    beta_prior_examples[2] = beta_prior_sample + (total1[idx2] - blue1[idx2]) + (total2[idx2] - blue2[idx2]);
    
    alpha_prior_examples[3] = alpha_prior_sample + blue1[idx3] + blue2[idx3];
    beta_prior_examples[3] = beta_prior_sample + (total1[idx3] - blue1[idx3]) + (total2[idx3] - blue2[idx3]);
  }
}

