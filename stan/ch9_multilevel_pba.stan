
// Multilevel Proportional Bayesian Agent
// Population distribution on logit(p).
// Non-centred parameterisation for individual p_j.
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
  real mu_logit_p;             // population mean on logit scale
  real<lower=0> sigma_logit_p; // population SD on logit scale
  vector[J] z_p;               // NCP deviates
}

transformed parameters {
  vector<lower=0, upper=1>[J] p;
  for (j in 1:J)
    p[j] = inv_logit(mu_logit_p + sigma_logit_p * z_p[j]);
}

model {
  target += normal_lpdf(mu_logit_p    | 0, 1.5);
  target += exponential_lpdf(sigma_logit_p | 2);
  target += std_normal_lpdf(z_p);
  
  for (i in 1:N) {
    int j = agent_id[i];
    real alpha_post = 0.5 + p[j] * blue1[i] + (1.0 - p[j]) * blue2[i];
    real beta_post  = 0.5 + p[j] * (total1[i] - blue1[i])
                          + (1.0 - p[j]) * (total2[i] - blue2[i]);
    target += beta_binomial_lpmf(choice[i] | 1, alpha_post, beta_post);
  }
}

generated quantities {
  real pop_p = inv_logit(mu_logit_p);
  
  vector[N] log_lik;
  for (i in 1:N) {
    int j = agent_id[i];
    real alpha_post = 0.5 + p[j] * blue1[i] + (1.0 - p[j]) * blue2[i];
    real beta_post  = 0.5 + p[j] * (total1[i] - blue1[i])
                          + (1.0 - p[j]) * (total2[i] - blue2[i]);
    log_lik[i] = beta_binomial_lpmf(choice[i] | 1, alpha_post, beta_post);
  }
}

