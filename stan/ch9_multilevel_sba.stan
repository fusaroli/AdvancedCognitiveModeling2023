
// Multilevel Simple Bayesian Agent
// Agents vary in an overall evidence scaling factor (mu_log_scale),
// but both sources always receive the same weight.
data {
  int<lower=1> N;
  int<lower=1> J;
  array[N] int<lower=1, upper=J> agent_id;
  array[N] int<lower=0, upper=1>  choice;
  array[N] int<lower=0>            blue1;
  array[N] int<lower=0>            total1;
  array[N] int<lower=0>            blue2;
  array[N] int<lower=0>            total2;
}

parameters {
  real mu_log_scale;             // Population mean log scale
  real<lower=0> sigma_log_scale; // Between-subject variability
  vector[J] z_scale;             // NCP z-scores
}

transformed parameters {
  vector<lower=0>[J] scale;
  for (j in 1:J)
    scale[j] = exp(mu_log_scale + z_scale[j] * sigma_log_scale);
}

model {
  target += normal_lpdf(mu_log_scale    | 0, 1);
  target += exponential_lpdf(sigma_log_scale | 2);
  target += std_normal_lpdf(z_scale);

  for (i in 1:N) {
    int j = agent_id[i];
    // Equal weights: both sources scaled identically
    real alpha_post = 1.0 + scale[j] * (blue1[i]  + blue2[i]);
    real beta_post  = 1.0 + scale[j] * ((total1[i] - blue1[i]) +
                                         (total2[i] - blue2[i]));
    target += beta_binomial_lpmf(choice[i] | 1, alpha_post, beta_post);
  }
}

generated quantities {
  real pop_scale = exp(mu_log_scale);
  vector[N] log_lik;
  array[N] int pred_choice;

  for (i in 1:N) {
    int j = agent_id[i];
    real alpha_post = 1.0 + scale[j] * (blue1[i]  + blue2[i]);
    real beta_post  = 1.0 + scale[j] * ((total1[i] - blue1[i]) +
                                         (total2[i] - blue2[i]));
    log_lik[i]     = beta_binomial_lpmf(choice[i] | 1, alpha_post, beta_post);
    pred_choice[i] = beta_binomial_rng(1, alpha_post, beta_post);
  }
}

