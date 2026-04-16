
// Circular Inference Agent (Jardri & Deneve 2013; Jardri et al. 2017).
// Bernoulli likelihood (not BetaBinomial) — see observation model discussion.
data {
  int<lower=1> N;
  array[N] int<lower=0, upper=1> choice;
  array[N] int<lower=0> blue1;
  array[N] int<lower=0> total1;
  array[N] int<lower=0> blue2;
  array[N] int<lower=0> total2;
}

transformed data {
  real alpha0 = 0.5;
  real beta0  = 0.5;
  real lo_prior = log(alpha0 / beta0);
}

parameters {
  real<lower=0> w_self;
  real<lower=0> w_other;
  real<lower=0> alpha_self_m1;    // excess over 1: alpha_self = 1 + alpha_self_m1
  real<lower=0> alpha_other_m1;
}

transformed parameters {
  real<lower=1> alpha_self  = 1.0 + alpha_self_m1;
  real<lower=1> alpha_other = 1.0 + alpha_other_m1;
}

model {
  target += lognormal_lpdf(w_self  | 0, 0.5);
  target += lognormal_lpdf(w_other | 0, 0.5);
  // Half-normal prior on excess loop strength (alpha >= 1 by construction)
  target += normal_lpdf(alpha_self_m1  | 0, 1) - normal_lccdf(0 | 0, 1);
  target += normal_lpdf(alpha_other_m1 | 0, 1) - normal_lccdf(0 | 0, 1);

  for (i in 1:N) {
    real lo_self_i   = log((alpha0 + blue1[i]) /
                           (beta0  + total1[i] - blue1[i]));
    real lo_social_i = log((alpha0 + blue2[i]) /
                           (beta0  + total2[i] - blue2[i]));
    real lo_combined = w_self  * alpha_self  * (lo_self_i   - lo_prior) +
                       w_other * alpha_other * (lo_social_i - lo_prior) +
                       lo_prior;
    target += bernoulli_logit_lpmf(choice[i] | lo_combined);
  }
}

generated quantities {
  vector[N] log_lik;
  array[N] int prior_pred;
  array[N] int posterior_pred;

  // Prior draws for prior predictive checks
  real ws_pr  = lognormal_rng(0, 0.5);
  real wo_pr  = lognormal_rng(0, 0.5);
  real as_pr  = 1.0 + abs(normal_rng(0, 1));
  real ao_pr  = 1.0 + abs(normal_rng(0, 1));

  for (i in 1:N) {
    real lo_self_i   = log((alpha0 + blue1[i]) /
                           (beta0  + total1[i] - blue1[i]));
    real lo_social_i = log((alpha0 + blue2[i]) /
                           (beta0  + total2[i] - blue2[i]));

    real lo_post = w_self  * alpha_self  * (lo_self_i   - lo_prior) +
                   w_other * alpha_other * (lo_social_i - lo_prior) +
                   lo_prior;

    log_lik[i]        = bernoulli_logit_lpmf(choice[i] | lo_post);
    posterior_pred[i] = bernoulli_logit_rng(lo_post);

    real lo_prior_pred = ws_pr * as_pr * (lo_self_i   - lo_prior) +
                         wo_pr * ao_pr * (lo_social_i - lo_prior) +
                         lo_prior;
    prior_pred[i] = bernoulli_logit_rng(lo_prior_pred);
  }
}

