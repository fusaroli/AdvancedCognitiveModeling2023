
functions {
  // The core Jardri transformation function
  real jardri_f(real L, real w) {
    real num = w * exp(L) + (1.0 - w);
    real den = (1.0 - w) * exp(L) + w;
    return log(num / den);
  }
}

data {
  int<lower=1> N_total;
  array[N_total] int<lower=0, upper=1> y;
  array[N_total] int<lower=0> blue1;
  array[N_total] int<lower=0> total1;
  array[N_total] int<lower=0> blue2;
  array[N_total] int<lower=0> total2;

  real prior_w_mu;
  real<lower=0> prior_w_sigma;
  int<lower=0, upper=1> run_diagnostics;
}

transformed data {
  // Convert raw counts to log-odds once to save computation
  real alpha0 = 0.5;
  real beta0  = 0.5;
  array[N_total] real lo_self;
  array[N_total] real lo_social;

  for (i in 1:N_total) {
    lo_self[i]   = log((alpha0 + blue1[i]) / (beta0 + total1[i] - blue1[i]));
    lo_social[i] = log((alpha0 + blue2[i]) / (beta0 + total2[i] - blue2[i]));
  }
}

parameters {
  // Scaled between 0.5 (no trust) and 1.0 (full trust)
  real<lower=0.5, upper=1.0> w_self;
  real<lower=0.5, upper=1.0> w_other;
}

model {
  // Weakly informative priors centered between 0.5 and 1.0
  target += normal_lpdf(w_self | prior_w_mu, prior_w_sigma);
  target += normal_lpdf(w_other | prior_w_mu, prior_w_sigma);

  for (i in 1:N_total) {
    real L_post = jardri_f(lo_self[i], w_self) + 
                  jardri_f(lo_social[i], w_other);
    target += bernoulli_logit_lpmf(y[i] | L_post);
  }
}

generated quantities {
  vector[N_total] log_lik;
  array[N_total] int y_rep;

  if (run_diagnostics) {
    for (i in 1:N_total) {
      real L_post = jardri_f(lo_self[i], w_self) + 
                    jardri_f(lo_social[i], w_other);
      log_lik[i] = bernoulli_logit_lpmf(y[i] | L_post);
      y_rep[i]   = bernoulli_logit_rng(L_post);
    }
  }
}

