
// Multilevel Simple Bayesian Agent
// Zero free parameters per agent. Strict integration applied uniformly.
data {
  int<lower=1> N_total;
  int<lower=1> N_subjects;
  array[N_total] int<lower=1, upper=N_subjects> subj_id;
  array[N_total] int<lower=0, upper=1>  y;
  array[N_total] int<lower=0>            blue1;
  array[N_total] int<lower=0>            total1;
  array[N_total] int<lower=0>            blue2;
  array[N_total] int<lower=0>            total2;
  int<lower=0, upper=1> run_diagnostics;
}

transformed data {
  real alpha0 = 0.5;
  real beta0  = 0.5;
}

model {
  vector[N_total] alpha_post = alpha0 + to_vector(blue1) + to_vector(blue2);
  vector[N_total] beta_post  = beta0  + (to_vector(total1) - to_vector(blue1)) +
                                        (to_vector(total2) - to_vector(blue2));
  target += beta_binomial_lpmf(y | 1, alpha_post, beta_post);
}

generated quantities {
  vector[N_total] log_lik;
  array[N_total] int y_rep;

  if (run_diagnostics) {
    for (i in 1:N_total) {
      real alpha_post = alpha0 + blue1[i] + blue2[i];
      real beta_post  = beta0 + (total1[i] - blue1[i]) + (total2[i] - blue2[i]);
      log_lik[i] = beta_binomial_lpmf(y[i] | 1, alpha_post, beta_post);
      y_rep[i]   = beta_binomial_rng(1, alpha_post, beta_post);
    }
  }
}

