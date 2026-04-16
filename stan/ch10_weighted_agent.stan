
// Weighted Bayesian Agent (reparameterised).
// rho in (0,1): relative weight of direct vs social evidence.
// kappa > 0: total evidence scaling (w_d + w_s).
data {
  int<lower=1> N;
  array[N] int<lower=0, upper=1> choice;
  array[N] int<lower=0> blue1;
  array[N] int<lower=0> total1;
  array[N] int<lower=0> blue2;
  array[N] int<lower=0> total2;
}

parameters {
  real<lower=0, upper=1> rho;    // relative weight: w_d / (w_d + w_s)
  real<lower=0>          kappa;  // total weight: w_d + w_s
}

transformed parameters {
  real<lower=0> weight_direct = rho * kappa;
  real<lower=0> weight_social = (1.0 - rho) * kappa;
}

model {
  // rho: weakly centred on equal weighting
  target += beta_lpdf(rho | 2, 2);
  // kappa: lognormal centered on 2 (SBA equivalent)
  target += lognormal_lpdf(kappa | log(2), 0.5);

  // Vectorized likelihood
  vector[N] alpha_post = 0.5 + weight_direct * to_vector(blue1)
                             + weight_social * to_vector(blue2);
  vector[N] beta_post  = 0.5 + weight_direct * (to_vector(total1) - to_vector(blue1))
                             + weight_social * (to_vector(total2) - to_vector(blue2));
                             
  target += beta_binomial_lpmf(choice | 1, alpha_post, beta_post);
}

generated quantities {
  vector[N] log_lik;
  array[N] int prior_pred;
  array[N] int posterior_pred;
  real lprior = beta_lpdf(rho | 2, 2) + lognormal_lpdf(kappa | log(2), 0.5);

  real rho_prior   = beta_rng(2, 2);
  real kappa_prior = lognormal_rng(log(2), 0.5);
  real wd_prior    = rho_prior * kappa_prior;
  real ws_prior    = (1.0 - rho_prior) * kappa_prior;

  for (i in 1:N) {
    real alpha_post = 0.5 + weight_direct * blue1[i] + weight_social * blue2[i];
    real beta_post  = 0.5 + weight_direct * (total1[i] - blue1[i]) 
                         + weight_social * (total2[i] - blue2[i]);

    log_lik[i]        = beta_binomial_lpmf(choice[i] | 1, alpha_post, beta_post);
    posterior_pred[i] = beta_binomial_rng(1, alpha_post, beta_post);

    real ap = 0.5 + wd_prior * blue1[i] + ws_prior * blue2[i];
    real bp = 0.5 + wd_prior * (total1[i] - blue1[i]) 
                  + ws_prior * (total2[i] - blue2[i]);
    prior_pred[i] = beta_binomial_rng(1, ap, bp);
  }
}

