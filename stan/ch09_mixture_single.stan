
data {
  int<lower=1> n;
  array[n] int<lower=0, upper=1> h;
}

parameters {
  real alpha;        // logit-scale bias (biased component)
  real nu;           // logit-scale mixing weight (random component)
}

model {
  real noise_p = inv_logit(nu);

  target += normal_lpdf(alpha | 0, 1);
  target += normal_lpdf(nu    | -1, 1);

  // Correct: one log_mix call per observation.
  // logit(0.5) = 0.0, so bernoulli_logit_lpmf(h[i] | 0.0) is used for
  // the random component — keeping both calls on the logit scale for
  // consistency and to avoid an implicit probability-to-logit conversion.
  for (i in 1:n)
    target += log_mix(noise_p,
                      bernoulli_logit_lpmf(h[i] | 0.0),    // random component
                      bernoulli_logit_lpmf(h[i] | alpha));  // biased component
}

generated quantities {
  real<lower=0, upper=1> theta   = inv_logit(alpha);  // bias, probability scale
  real<lower=0, upper=1> pi_hat  = inv_logit(nu);     // noise weight, prob scale

  // lprior is required by priorsense::powerscale_sensitivity()
  real lprior = normal_lpdf(alpha | 0, 1) + normal_lpdf(nu | -1, 1);

  vector[n] log_lik;
  array[n] int pred_choice;

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

