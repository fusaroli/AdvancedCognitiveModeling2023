
// Weighted Bayesian Agent.
// w_direct, w_social >= 0 scale the effective count of each evidence source.
data {
  int<lower=1> N;
  array[N] int<lower=0, upper=1> choice;
  array[N] int<lower=0> blue1;
  array[N] int<lower=0> total1;
  array[N] int<lower=0> blue2;
  array[N] int<lower=0> total2;
}

parameters {
  real<lower=0> weight_direct;
  real<lower=0> weight_social;
}

model {
  // Priors on log scale (equivalent to log-normal in natural scale).
  // lognormal(0, 0.5) concentrates mass roughly in [0.2, 5],
  // spanning strong underweighting to strong overweighting.
  target += lognormal_lpdf(weight_direct | 0, 0.5);
  target += lognormal_lpdf(weight_social | 0, 0.5);

  profile("likelihood") {
    for (i in 1:N) {
      real alpha_post = 1.0
                      + weight_direct * blue1[i]
                      + weight_social * blue2[i];
      real beta_post  = 1.0
                      + weight_direct * (total1[i] - blue1[i])
                      + weight_social * (total2[i] - blue2[i]);
      target += beta_binomial_lpmf(choice[i] | 1, alpha_post, beta_post);
    }
  }
}

generated quantities {
  vector[N] log_lik;
  array[N] int prior_pred;
  array[N] int posterior_pred;

  // Prior samples for predictive checks
  real wd_prior = lognormal_rng(0, 0.5);
  real ws_prior = lognormal_rng(0, 0.5);

  for (i in 1:N) {
    // Posterior predictions
    real alpha_post = 1.0
                    + weight_direct * blue1[i]
                    + weight_social * blue2[i];
    real beta_post  = 1.0
                    + weight_direct * (total1[i] - blue1[i])
                    + weight_social * (total2[i] - blue2[i]);

    log_lik[i]        = beta_binomial_lpmf(choice[i] | 1, alpha_post, beta_post);
    posterior_pred[i] = beta_binomial_rng(1, alpha_post, beta_post);

    // Prior predictions using sampled prior weights
    real ap = 1.0 + wd_prior * blue1[i] + ws_prior * blue2[i];
    real bp = 1.0 + wd_prior * (total1[i] - blue1[i])
                  + ws_prior * (total2[i] - blue2[i]);
    prior_pred[i] = beta_binomial_rng(1, ap, bp);
  }
}

