
data {
  int<lower=1> n;  // Number of trials
  array[n] int h;  // Choice data (0/1)
}

parameters {
  real bias;  // Bias parameter (logit scale)
}

model {
  // Prior
  target += normal_lpdf(bias | 0, 1);
  
  // Likelihood (all choices come from biased process)
  target += bernoulli_logit_lpmf(h | bias);
}

generated quantities {
  real<lower=0, upper=1> bias_p = inv_logit(bias);  // Bias on probability scale
  
  // Log likelihood for model comparison
  vector[n] log_lik;
  for (i in 1:n) {
    log_lik[i] = bernoulli_logit_lpmf(h[i] | bias);
  }
  
  // Posterior predictions
  array[n] int pred_choice;
  for (i in 1:n) {
    pred_choice[i] = bernoulli_rng(bias_p);
  }
}

