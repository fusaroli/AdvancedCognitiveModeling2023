
// Sequential Bayesian Updating Model
// This model tracks how an agent updates beliefs across a sequence of trials
data {
  int<lower=1> T;                        // Number of trials
  array[T] int<lower=0, upper=1> choice; // Choices (0=red, 1=blue)
  array[T] int<lower=0> blue1;           // Direct evidence (blue marbles) on each trial
  array[T] int<lower=0> total1;          // Total direct evidence on each trial
  array[T] int<lower=0> blue2;           // Social evidence (blue signals) on each trial
  array[T] int<lower=0> total2;          // Total social evidence on each trial
}

parameters {
  real<lower=0> total_weight;             // Overall weight given to evidence
  real<lower=0, upper=1> weight_prop;     // Proportion of weight for direct evidence
  real<lower=0> alpha;                    // Learning rate parameter
}

transformed parameters {
  // Calculate weights for each evidence source
  real weight_direct = total_weight * weight_prop;
  real weight_social = total_weight * (1 - weight_prop);
  
  // Variables to track belief updating across trials
  vector<lower=0, upper=1>[T] belief;     // Belief in blue on each trial
  vector<lower=0>[T] alpha_param;         // Beta distribution alpha parameter
  vector<lower=0>[T] beta_param;          // Beta distribution beta parameter
  
  // Initial belief parameters (uniform prior)
  alpha_param[1] = 1.0;
  beta_param[1] = 1.0;
  
  // Calculate belief for first trial
  belief[1] = alpha_param[1] / (alpha_param[1] + beta_param[1]);
  
  // Update beliefs across trials
  for (t in 2:T) {
    // Calculate weighted evidence from previous trial
    real weighted_blue1 = blue1[t-1] * weight_direct;
    real weighted_red1 = (total1[t-1] - blue1[t-1]) * weight_direct;
    real weighted_blue2 = blue2[t-1] * weight_social;
    real weighted_red2 = (total2[t-1] - blue2[t-1]) * weight_social;
    
    // Update belief with learning rate
    // alpha controls how much new evidence affects the belief
    alpha_param[t] = alpha_param[t-1] + alpha * (weighted_blue1 + weighted_blue2);
    beta_param[t] = beta_param[t-1] + alpha * (weighted_red1 + weighted_red2);
    
    // Calculate updated belief
    belief[t] = alpha_param[t] / (alpha_param[t] + beta_param[t]);
  }
}

model {
  // Priors for parameters
  target += lognormal_lpdf(total_weight | 0, 0.5);  // Prior centered around 1.0
  target += beta_lpdf(weight_prop | 1, 1);          // Uniform prior on proportion
  target += lognormal_lpdf(alpha | -1, 0.5);        // Prior on learning rate (typically < 1)
  
  // Likelihood
  for (t in 1:T) {
    // Model choice as a function of current belief
    target += bernoulli_lpmf(choice[t] | belief[t]);
  }
}

generated quantities {
  // Log likelihood for model comparison
  vector[T] log_lik;
  
  // Posterior predictions
  array[T] int pred_choice;
  
  for (t in 1:T) {
    // Generate predicted choices
    pred_choice[t] = bernoulli_rng(belief[t]);
    
    // Calculate log likelihood
    log_lik[t] = bernoulli_lpmf(choice[t] | belief[t]);
  }
}

