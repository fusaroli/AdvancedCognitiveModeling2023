
data {
  int<lower=1> n;
  array[n] int<lower=0, upper=1> h;
  real prior_alpha_mu;
  real<lower=0> prior_alpha_sigma;
  real prior_nu_mu;
  real<lower=0> prior_nu_sigma;
  int<lower=0, upper=1> run_diagnostics;
}

parameters {
  real alpha;        // logit-scale bias (biased component)
  real nu;           // logit-scale mixing weight (random component)
}

model {
  real noise_p = inv_logit(nu);

  target += normal_lpdf(alpha | prior_alpha_mu, prior_alpha_sigma);
  target += normal_lpdf(nu    | prior_nu_mu,    prior_nu_sigma);

  // Marginalise the discrete component indicator analytically via log_mix.
  // One call per observation — vectorisation is not available for log_mix.
  for (i in 1:n)
    target += log_mix(noise_p,
                      bernoulli_logit_lpmf(h[i] | 0.0),
                      bernoulli_logit_lpmf(h[i] | alpha));
}

generated quantities {
  real<lower=0, upper=1> theta   = inv_logit(alpha);
  real<lower=0, upper=1> pi_hat  = inv_logit(nu);
  real lprior = normal_lpdf(alpha | prior_alpha_mu, prior_alpha_sigma)
              + normal_lpdf(nu    | prior_nu_mu,    prior_nu_sigma);
  vector[n] log_lik;
  array[n] int pred_choice;
  if (run_diagnostics) {
    for (i in 1:n) {
      log_lik[i] = log_mix(pi_hat,
                           bernoulli_logit_lpmf(h[i] | 0.0),
                           bernoulli_logit_lpmf(h[i] | alpha));
      if (bernoulli_rng(pi_hat))
        pred_choice[i] = bernoulli_rng(0.5);
      else
        pred_choice[i] = bernoulli_rng(theta);
    }
  }
}

