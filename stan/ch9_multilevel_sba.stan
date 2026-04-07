
// Multilevel Simple Bayesian Agent
// Zero free parameters per agent. Strict integration applied uniformly.
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

transformed data {
  real alpha0 = 0.5;
  real beta0  = 0.5;
}

model {
  vector[N] alpha_post = alpha0 + to_vector(blue1) + to_vector(blue2);
  vector[N] beta_post  = beta0  + (to_vector(total1) - to_vector(blue1)) +
                                  (to_vector(total2) - to_vector(blue2));
  target += beta_binomial_lpmf(choice | 1, alpha_post, beta_post);
}

generated quantities {
  vector[N] log_lik;
  for (i in 1:N) {
    real alpha_post = alpha0 + blue1[i] + blue2[i];
    real beta_post  = beta0 + (total1[i] - blue1[i]) + (total2[i] - blue2[i]);
    log_lik[i] = beta_binomial_lpmf(choice[i] | 1, alpha_post, beta_post);
  }
}

