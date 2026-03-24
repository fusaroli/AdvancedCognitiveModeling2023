
// Circular Inference Agent (Jardri & Deneve 2013; Jardri et al. 2017).
// Works on the log-odds scale.
// Parameters:
//   w_self, w_other    : trust weights (positive; <1 = discount, >1 = overtrust)
//   alpha_self, alpha_other : loop / overcounting parameters (>= 1 for CI)
// When alpha = 1 and w = 1 for both sources, reduces to simple Bayes.
// When alpha = 1, reduces to Weighted Bayes on the log-odds scale.
//
// NOTE: The likelihood is Bernoulli, not BetaBinomial.
// The BetaBinomial models explicit Beta uncertainty about theta and is natural
// in the pseudocount framework. Here beliefs are represented as a single
// log-odds value; the Bernoulli likelihood is the appropriate observation model.
data {
  int<lower=1> N;
  array[N] int<lower=0, upper=1> choice;
  array[N] int<lower=0> blue1;
  array[N] int<lower=0> total1;   // = 8 throughout
  array[N] int<lower=0> blue2;
  array[N] int<lower=0> total2;   // = 3 throughout
}

transformed data {
  // Jeffreys prior pseudo-counts — same as in the WBA for comparability
  real alpha0 = 0.5;
  real beta0  = 0.5;
  // Log-prior log-odds (= 0 for Jeffreys since alpha0 = beta0)
  real lo_prior = log(alpha0 / beta0);
}

parameters {
  // Trust weights: positive, log-normal prior matching the WBA
  real<lower=0> w_self;
  real<lower=0> w_other;

  // Loop parameters: >= 1. Parameterised as 1 + exp(raw) so the unconstrained
  // sampler works on an unbounded real; alpha = 1 corresponds to raw = -Inf
  // (standard Bayes). We use a weakly regularising half-normal prior on (alpha - 1).
  real<lower=0> alpha_self_m1;   // alpha_self  = 1 + alpha_self_m1
  real<lower=0> alpha_other_m1;  // alpha_other = 1 + alpha_other_m1
}

transformed parameters {
  real<lower=1> alpha_self  = 1.0 + alpha_self_m1;
  real<lower=1> alpha_other = 1.0 + alpha_other_m1;
}

model {
  // Priors
  // Weights: lognormal(0, 0.5) as in WBA — spans underweighting to overweighting
  target += lognormal_lpdf(w_self  | 0, 0.5);
  target += lognormal_lpdf(w_other | 0, 0.5);

  // Loop parameters: half-normal(0, 1) on the excess above 1.
  // Concentrates mass near alpha = 1 (no overcounting) with moderate regularisation.
  // A value of alpha_self_m1 ~ 2 implies alpha_self ~ 3, which is already strong CI.
  target += normal_lpdf(alpha_self_m1  | 0, 1) - normal_lccdf(0 | 0, 1);
  target += normal_lpdf(alpha_other_m1 | 0, 1) - normal_lccdf(0 | 0, 1);

  // Likelihood
  for (i in 1:N) {
    // Log-likelihood ratios (log-odds of each source relative to prior)
    real lo_self_i   = log((alpha0 + blue1[i]) /
                           (beta0  + total1[i] - blue1[i]));
    real lo_social_i = log((alpha0 + blue2[i]) /
                           (beta0  + total2[i] - blue2[i]));

    // Circular-inference combined log-odds
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

  // Prior samples for predictive checks
  real ws_pr  = lognormal_rng(0, 0.5);
  real wo_pr  = lognormal_rng(0, 0.5);
  real as_pr  = 1.0 + abs(normal_rng(0, 1));
  real ao_pr  = 1.0 + abs(normal_rng(0, 1));

  for (i in 1:N) {
    real lo_self_i   = log((alpha0 + blue1[i]) /
                           (beta0  + total1[i] - blue1[i]));
    real lo_social_i = log((alpha0 + blue2[i]) /
                           (beta0  + total2[i] - blue2[i]));

    // Posterior
    real lo_post = w_self  * alpha_self  * (lo_self_i   - lo_prior) +
                   w_other * alpha_other * (lo_social_i - lo_prior) +
                   lo_prior;

    log_lik[i]        = bernoulli_logit_lpmf(choice[i] | lo_post);
    posterior_pred[i] = bernoulli_logit_rng(lo_post);

    // Prior predictive
    real lo_prior_pred = ws_pr * as_pr * (lo_self_i   - lo_prior) +
                         wo_pr * ao_pr * (lo_social_i - lo_prior) +
                         lo_prior;
    prior_pred[i] = bernoulli_logit_rng(lo_prior_pred);
  }
}

