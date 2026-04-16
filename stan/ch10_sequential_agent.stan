
// Sequential Bayesian Updating Model
// lambda (learning_rate) scales how much each trial's evidence is accumulated.
data {
  int<lower=1> T;
  array[T] int<lower=0, upper=1> choice;
  array[T] int<lower=0> blue1;
  array[T] int<lower=0> total1;
  array[T] int<lower=0> blue2;
  array[T] int<lower=0> total2;
}

parameters {
  real<lower=0> weight_direct;
  real<lower=0> weight_social;
  real<lower=0> learning_rate;   // consistent name throughout
}

transformed parameters {
  vector<lower=0, upper=1>[T] belief;
  vector<lower=0>[T] alpha_seq;
  vector<lower=0>[T] beta_seq;

  alpha_seq[1] = 1.0;
  beta_seq[1]  = 1.0;
  belief[1]    = alpha_seq[1] / (alpha_seq[1] + beta_seq[1]);

  for (t in 2:T) {
    real w_blue1 = blue1[t-1]  * weight_direct;
    real w_red1  = (total1[t-1] - blue1[t-1])  * weight_direct;
    real w_blue2 = blue2[t-1]  * weight_social;
    real w_red2  = (total2[t-1] - blue2[t-1])  * weight_social;

    alpha_seq[t] = alpha_seq[t-1] + learning_rate * (w_blue1 + w_blue2);
    beta_seq[t]  = beta_seq[t-1]  + learning_rate * (w_red1  + w_red2);
    belief[t]    = alpha_seq[t]   / (alpha_seq[t] + beta_seq[t]);
  }
}

model {
  target += lognormal_lpdf(weight_direct | 0, 0.5);
  target += lognormal_lpdf(weight_social | 0, 0.5);
  target += lognormal_lpdf(learning_rate | -1, 0.5);  // prior concentrated < 1

  for (t in 1:T) {
    target += bernoulli_lpmf(choice[t] | belief[t]);
  }
}

generated quantities {
  vector[T] log_lik;
  array[T] int pred_choice;

  for (t in 1:T) {
    log_lik[t]    = bernoulli_lpmf(choice[t] | belief[t]);
    pred_choice[t] = bernoulli_rng(belief[t]);
  }
}
// NOTE: Do NOT use loo() on log_lik from this model.
// Observations are not exchangeable. Use loo::loo_approximate_leave_future_out()
// (Burkner et al. 2020) for valid out-of-sample evaluation.

