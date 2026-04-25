
// Multilevel Proportional Bayesian Agent
// Population distribution on logit(p).
// Non-centred parameterisation for individual p_j.
data {
  int<lower=1> N_total;
  int<lower=1> N_subjects;
  array[N_total] int<lower=1, upper=N_subjects> subj_id;
  array[N_total] int<lower=0, upper=1>  y;
  array[N_total] int<lower=0>            blue1;
  array[N_total] int<lower=0>            total1;
  array[N_total] int<lower=0>            blue2;
  array[N_total] int<lower=0>            total2;
  
  real prior_mu_logit_p_mu;
  real<lower=0> prior_mu_logit_p_sigma;
  real<lower=0> prior_sigma_logit_p_lambda;
  int<lower=0, upper=1> run_diagnostics;
}

parameters {
  real mu_logit_p;             // population mean on logit scale
  real<lower=0> sigma_logit_p; // population SD on logit scale
  vector[N_subjects] z_p;      // NCP deviates
}

transformed parameters {
  vector<lower=0, upper=1>[N_subjects] p;
  for (j in 1:N_subjects)
    p[j] = inv_logit(mu_logit_p + sigma_logit_p * z_p[j]);
}

model {
  target += normal_lpdf(mu_logit_p    | prior_mu_logit_p_mu, prior_mu_logit_p_sigma);
  target += exponential_lpdf(sigma_logit_p | prior_sigma_logit_p_lambda);
  target += std_normal_lpdf(z_p);
  
  for (i in 1:N_total) {
    int j = subj_id[i];
    real alpha_post = 0.5 + p[j] * blue1[i] + (1.0 - p[j]) * blue2[i];
    real beta_post  = 0.5 + p[j] * (total1[i] - blue1[i])
                          + (1.0 - p[j]) * (total2[i] - blue2[i]);
    target += beta_binomial_lpmf(y[i] | 1, alpha_post, beta_post);
  }
}

generated quantities {
  real pop_p = inv_logit(mu_logit_p);
  vector[N_total] log_lik;
  array[N_total] int y_rep;

  if (run_diagnostics) {
    for (i in 1:N_total) {
      int j = subj_id[i];
      real alpha_post = 0.5 + p[j] * blue1[i] + (1.0 - p[j]) * blue2[i];
      real beta_post  = 0.5 + p[j] * (total1[i] - blue1[i])
                            + (1.0 - p[j]) * (total2[i] - blue2[i]);
      log_lik[i] = beta_binomial_lpmf(y[i] | 1, alpha_post, beta_post);
      y_rep[i]   = beta_binomial_rng(1, alpha_post, beta_post);
    }
  }
}

