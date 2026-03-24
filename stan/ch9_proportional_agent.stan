
// Proportional Bayesian Agent (PBA).
// p in [0,1] allocates the unit evidence budget between direct and social.
// p = 0.5 approximates the SBA; p -> 1 ignores social; p -> 0 ignores direct.
// Jeffreys prior pseudo-counts (alpha0 = beta0 = 0.5) consistent with SBA.
data {
  int<lower=1> N;
  array[N] int<lower=0, upper=1> choice;
  array[N] int<lower=0> blue1;
  array[N] int<lower=0> total1;
  array[N] int<lower=0> blue2;
  array[N] int<lower=0> total2;
}

parameters {
  // Allocation of the unit evidence budget to direct evidence.
  // Beta(2, 2): weakly bell-shaped, symmetric about 0.5, keeps
  // prior mass away from the boundaries where geometry can degrade.
  real<lower=0, upper=1> p;
}

model {
  target += beta_lpdf(p | 2, 2);

  profile("likelihood") {
    for (i in 1:N) {
      real alpha_post = 0.5
                      + p         * blue1[i]
                      + (1.0 - p) * blue2[i];
      real beta_post  = 0.5
                      + p         * (total1[i] - blue1[i])
                      + (1.0 - p) * (total2[i] - blue2[i]);
      target += beta_binomial_lpmf(choice[i] | 1, alpha_post, beta_post);
    }
  }
}

generated quantities {
  vector[N] log_lik;
  array[N] int prior_pred;
  array[N] int posterior_pred;

  real p_prior = beta_rng(2, 2);

  for (i in 1:N) {
    real alpha_post = 0.5 + p * blue1[i] + (1.0 - p) * blue2[i];
    real beta_post  = 0.5
                    + p         * (total1[i] - blue1[i])
                    + (1.0 - p) * (total2[i] - blue2[i]);

    log_lik[i]        = beta_binomial_lpmf(choice[i] | 1, alpha_post, beta_post);
    posterior_pred[i] = beta_binomial_rng(1, alpha_post, beta_post);

    // Prior predictive using sampled prior p
    real ap = 0.5 + p_prior * blue1[i] + (1.0 - p_prior) * blue2[i];
    real bp = 0.5
            + p_prior * (total1[i] - blue1[i])
            + (1.0 - p_prior) * (total2[i] - blue2[i]);
    prior_pred[i] = beta_binomial_rng(1, ap, bp);
  }
}

