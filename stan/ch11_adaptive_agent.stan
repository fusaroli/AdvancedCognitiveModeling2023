
// Adaptive-Weight Agent.
// Parameters: eta (learning rate), log w_d0, log w_s0 (initial log-weights).
// Weights are updated across trials by a delta rule on log-weights using
// per-source absolute prediction errors against the feedback.

data {
  int<lower=1> N;
  array[N] int<lower=0, upper=1> guess;
  array[N] int<lower=0, upper=1> correct;
  array[N] int<lower=0> k_d;
  array[N] int<lower=0> n_d;
  array[N] int<lower=0> k_s;
  array[N] int<lower=0> n_s;
}

parameters {
  real<lower=0, upper=1> eta;
  real log_w_d0;
  real log_w_s0;
}

transformed parameters {
  vector[N] p_c;
  {
    real log_w_d = log_w_d0;
    real log_w_s = log_w_s0;
    for (i in 1:N) {
      real w_d = exp(log_w_d);
      real w_s = exp(log_w_s);

      real alpha_d = 0.5 + k_d[i];
      real beta_d  = 0.5 + (n_d[i] - k_d[i]);
      real alpha_s = 0.5 + k_s[i];
      real beta_s  = 0.5 + (n_s[i] - k_s[i]);
      real p_d = alpha_d / (alpha_d + beta_d);
      real p_s = alpha_s / (alpha_s + beta_s);

      real alpha_c = 0.5 + w_d * k_d[i] + w_s * k_s[i];
      real beta_c  = 0.5 + w_d * (n_d[i] - k_d[i]) + w_s * (n_s[i] - k_s[i]);
      p_c[i] = alpha_c / (alpha_c + beta_c);

      real d_d = abs(correct[i] - p_d);
      real d_s = abs(correct[i] - p_s);
      log_w_d += eta * (d_s - d_d);
      log_w_s += eta * (d_d - d_s);
    }
  }
}

model {
  target += beta_lpdf(eta | 2, 5);
  target += normal_lpdf(log_w_d0 | 0, 1);
  target += normal_lpdf(log_w_s0 | 0, 1);

  target += bernoulli_lpmf(guess | p_c);
}

