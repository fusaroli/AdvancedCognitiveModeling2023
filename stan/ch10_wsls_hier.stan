
data {
  int<lower=1> N;         // total number of transitions
  int<lower=1> I;         // number of individuals
  array[N] int<lower=1, upper=I>  id;
  array[N] int<lower=1, upper=3>  rel_shift;
  array[N] int<lower=1, upper=3>  outcome;
}
parameters {
  // Population-level log-odds for [Stay, CW] vs CCW, by outcome
  array[3] vector[2] mu_pop;
  array[3] vector<lower=0>[2] tau_pop;
  // Non-centered individual deviations: z[i, outcome, logit_index]
  array[I, 3] vector[2] z_ind;
}
transformed parameters {
  // Individual-level simplex entries via softmax of logit-stick
  array[I, 3] simplex[3] theta_ind;
  for (i in 1:I) {
    for (o in 1:3) {
      vector[2] lam = mu_pop[o] + tau_pop[o] .* z_ind[i, o];
      theta_ind[i, o] = softmax(append_row(lam, 0.0));
    }
  }
}
model {
  for (o in 1:3) {
    mu_pop[o]  ~ normal(0, 1);
    tau_pop[o] ~ exponential(2);
  }
  for (i in 1:I) {
    for (o in 1:3) {
      z_ind[i, o] ~ std_normal();
    }
  }
  for (t in 1:N)
    rel_shift[t] ~ categorical(theta_ind[id[t], outcome[t]]);
}
generated quantities {
  vector[N] log_lik;
  for (t in 1:N)
    log_lik[t] = categorical_lpmf(rel_shift[t] |
                                   theta_ind[id[t], outcome[t]]);
}

