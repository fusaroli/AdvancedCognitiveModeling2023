
// Sample-or-Guess Agent with a soft certainty threshold.
// Single parameter: tau in (0, 0.5), the certainty threshold.
// The stop/continue decision is modelled with a logistic soft rule
// around tau with fixed slope sigma_tau.

data {
  int<lower=1> N;
  int<lower=1> T_max;
  array[N] int<lower=0, upper=T_max> T;
  array[N] int<lower=0, upper=1> guess;
  array[N, T_max] int<lower=0, upper=1> draws;
  int<lower=0> social_k;
  int<lower=0> social_n;
  real<lower=0> sigma_tau;
}

parameters {
  // tau ~ scaled Beta(2,2) on (0, 0.5)
  real<lower=0, upper=0.5> tau;
}

model {
  // Prior: Beta(2,2) on (0, 0.5) via change of variables -> tau * 2 ~ Beta(2,2)
  target += beta_lpdf(2 * tau | 2, 2) + log(2);

  for (i in 1:N) {
    real alpha_t = 0.5 + social_k;
    real beta_t  = 0.5 + (social_n - social_k);

    // Steps 1 .. T[i] - 1: the agent continued (soft prob of NOT stopping)
    // At step T[i]: the agent stopped (soft prob of stopping)
    // If T[i] == 0 (agent would stop before seeing any direct draw) we
    // only contribute the final guess.
    for (t in 1:T[i]) {
      int d = draws[i, t];
      alpha_t += d;
      beta_t  += (1 - d);
      real c_t = abs(alpha_t / (alpha_t + beta_t) - 0.5);
      real p_stop = inv_logit((c_t - tau) / sigma_tau);
      if (t < T[i]) {
        target += bernoulli_lpmf(0 | p_stop);  // continued
      } else {
        target += bernoulli_lpmf(1 | p_stop);  // stopped
      }
    }

    // Final guess given the posterior at stopping time
    {
      real alpha_final = 0.5 + social_k;
      real beta_final  = 0.5 + (social_n - social_k);
      for (t in 1:T[i]) {
        alpha_final += draws[i, t];
        beta_final  += (1 - draws[i, t]);
      }
      real p_blue = alpha_final / (alpha_final + beta_final);
      target += bernoulli_lpmf(guess[i] | p_blue);
    }
  }
}

