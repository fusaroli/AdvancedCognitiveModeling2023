
data {
  int<lower=1> N;
  array[N] int<lower=1, upper=3> action;
}
parameters {
  simplex[3] theta;
}
model {
  theta ~ dirichlet(rep_vector(2.0, 3));
  action ~ categorical(theta);
}
generated quantities {
  vector[N] log_lik;
  array[N] int action_rep;
  real lprior = dirichlet_lpdf(theta | rep_vector(2.0, 3));
  for (t in 1:N) {
    log_lik[t]    = categorical_lpmf(action[t] | theta);
    action_rep[t] = categorical_rng(theta);
  }
}

