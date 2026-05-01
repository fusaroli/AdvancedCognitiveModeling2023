
data {
  int<lower=1> N;                            // number of transitions
  array[N] int<lower=1, upper=3> rel_shift;  // relative shift (1=stay, 2=CW, 3=CCW)
  array[N] int<lower=1, upper=3> outcome;    // previous outcome (1=win, 2=lose, 3=draw)
}
parameters {
  array[3] simplex[3] theta_rel;  // theta_rel[outcome] = transition probs
}
model {
  // WSLS-informative priors
  theta_rel[1] ~ dirichlet([5.0, 1.0, 1.0]');  // Win  -> Stay
  theta_rel[2] ~ dirichlet([1.0, 5.0, 1.0]');  // Lose -> CW shift
  theta_rel[3] ~ dirichlet([2.0, 2.0, 2.0]');  // Draw -> uniform
  for (t in 1:N)
    rel_shift[t] ~ categorical(theta_rel[outcome[t]]);
}
generated quantities {
  vector[N] log_lik;
  array[N] int shift_rep;
  real lprior = dirichlet_lpdf(theta_rel[1] | [5.0, 1.0, 1.0]')
              + dirichlet_lpdf(theta_rel[2] | [1.0, 5.0, 1.0]')
              + dirichlet_lpdf(theta_rel[3] | [2.0, 2.0, 2.0]');
  for (t in 1:N) {
    log_lik[t]   = categorical_lpmf(rel_shift[t] | theta_rel[outcome[t]]);
    shift_rep[t] = categorical_rng(theta_rel[outcome[t]]);
  }
}

