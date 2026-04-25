
data {
  int<lower=1> N_total;
  array[N_total] int<lower=0, upper=1> y;
  array[N_total] int<lower=0, upper=1> other;
  
  real prior_log_sigma_mu;
  real<lower=0> prior_log_sigma_sigma;
  real prior_log_beta_mu;
  real<lower=0> prior_log_beta_sigma;
  real prior_bias_mu;
  real<lower=0> prior_bias_sigma;
  int<lower=0, upper=1> run_diagnostics;
}
parameters {
  real log_sigma;
  real log_beta;
  real bias;
}
transformed parameters {
  vector[N_total] dV;
  {
    real sigma = exp(log_sigma);
    real mu    = 0.0;
    real Sig   = 1.0;
    for (t in 1:N_total) {
      real s_mu = inv_logit(mu);
      real p_op = inv_logit(mu / sqrt(1.0 + 0.36 * Sig));
      dV[t] = 2.0 * p_op - 1.0;
      Sig = 1.0 / (1.0 / (Sig + sigma) + s_mu * (1.0 - s_mu));
      mu  = mu + Sig * (other[t] - s_mu);
    }
  }
}
model {
  target += normal_lpdf(log_sigma | prior_log_sigma_mu, prior_log_sigma_sigma);
  target += normal_lpdf(log_beta  | prior_log_beta_mu,  prior_log_beta_sigma);
  target += normal_lpdf(bias      | prior_bias_mu,      prior_bias_sigma);
  target += bernoulli_logit_lpmf(y | (dV + bias) / exp(log_beta));
}
generated quantities {
  vector[N_total] log_lik;
  array[N_total] int y_rep;
  real lprior =   normal_lpdf(log_sigma | prior_log_sigma_mu, prior_log_sigma_sigma)
                + normal_lpdf(log_beta  | prior_log_beta_mu,  prior_log_beta_sigma)
                + normal_lpdf(bias      | prior_bias_mu,      prior_bias_sigma);
  if (run_diagnostics) {
    for (t in 1:N_total) {
      log_lik[t] = bernoulli_logit_lpmf(y[t] | (dV[t] + bias) / exp(log_beta));
      y_rep[t]   = bernoulli_logit_rng((dV[t] + bias) / exp(log_beta));
    }
  }
}

