
data {
  int<lower=1> N;
  array[N] int<lower=1, upper=3> action;
  array[N] int<lower=1, upper=3> op_action;
}
transformed data {
  matrix[3, 3] payoff = [[0.0, -1.0,  1.0],
                          [1.0,  0.0, -1.0],
                          [-1.0, 1.0,  0.0]];
}
parameters {
  real log_sigma;
  real log_beta;
}
transformed parameters {
  matrix[N, 3] EU;  // expected utility: row t is the EU vector at trial t
  {
    vector[3] alpha = rep_vector(1.0, 3);
    real rho  = exp(-exp(log_sigma));
    real beta = exp(log_beta);
    for (t in 1:N) {
      vector[3] p_op = alpha / sum(alpha);
      EU[t] = to_row_vector(payoff * p_op) / beta;  // matrix-vector product
      alpha = rho * alpha + (1.0 - rho) * rep_vector(1.0, 3);
      alpha[op_action[t]] += 1.0;
    }
  }
}
model {
  log_sigma ~ normal(-1, 1);
  log_beta  ~ normal(-1, 1);
  for (t in 1:N)
    action[t] ~ categorical_logit(to_vector(EU[t]));
}
generated quantities {
  vector[N] log_lik;
  array[N] int action_rep;
  real lprior = normal_lpdf(log_sigma | -1, 1)
              + normal_lpdf(log_beta  | -1, 1);
  for (t in 1:N) {
    log_lik[t]    = categorical_logit_lpmf(action[t] | to_vector(EU[t]));
    action_rep[t] = categorical_logit_rng(to_vector(EU[t]));
  }
}

