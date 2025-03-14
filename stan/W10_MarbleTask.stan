
data {
  int<lower=0> N;                 // Number of observations
  array[N] int<lower=0, upper=1> choice;  // Binary choices (0 = Red, 1 = Blue)
  array[N] int<lower=0, upper=8> blue1;   // Direct evidence (number of blue marbles)
  array[N] int<lower=0, upper=3> blue2;   // Social evidence (confidence level)
  int<lower=0> total1;            // Total marbles in direct evidence
  int<lower=0> total2;            // Total levels in social evidence
}
transformed data {
  array[N] real p_blue1;
  array[N] real p_blue2;
  for (n in 1:N) {
    // Convert to probability scale with pseudocounts for stability
    p_blue1[n] = (blue1[n] + 1.0) / (total1 + 2.0);
    p_blue2[n] = (blue2[n] + 1.0) / (total2 + 2.0);
  }
}
parameters {
  real bias;  // Bias parameter in log-odds scale
}
model {
  // Prior for bias
  target += normal_lpdf(bias | 0, 1);
  // Likelihood
  for (n in 1:N) {
    real belief = bias + logit(p_blue1[n]) + logit(p_blue2[n]);
    target += bernoulli_logit_lpmf(choice[n] | belief);
  }
}
generated quantities {
  // Prior predictive sample
  real bias_prior = normal_rng(0, 1);
  // Log-likelihood for model comparison
  array[N] real log_lik;
  // Posterior predictive samples
  array[N] real<lower=0, upper=1> predicted_belief;
  array[N] int<lower=0, upper=1> predicted_choice;
  for (n in 1:N) {
    real belief_logodds = bias + logit(p_blue1[n]) + logit(p_blue2[n]);
    log_lik[n] = bernoulli_logit_lpmf(choice[n] | belief_logodds);
    // Generate predictions
    predicted_belief[n] = inv_logit(belief_logodds);
    predicted_choice[n] = bernoulli_rng(predicted_belief[n]);
  }
}

