
data {
  int<lower=1> ntrials;
  int<lower=1> nfeatures;
  array[ntrials] int<lower=0, upper=1> y;
  array[ntrials, nfeatures] real obs;
  real prior_logit_error_mean;
  real<lower=0> prior_logit_error_sd;
}
parameters { real logit_error_prob; }
transformed parameters {
  real<lower=0, upper=1> error_prob = inv_logit(logit_error_prob);
  array[ntrials] real lp_marginal;
  for (i in 1:ntrials) {
    real h = obs[i, 1];
    real p = obs[i, 2];
    int pred1 = h > 4.5 ? 1 : 0;
    int pred2 = p > 4.5 ? 1 : 0;
    int pred3 = h < 0.5 ? 1 : 0;
    int pred4 = (h > 0.5 && p > 0.5) ? 1 : 0;
    int pred5 = (h > 0.5 || p > 0.5) ? 1 : 0;
    int R = 5;
    vector[R] lp;
    lp[1] = bernoulli_lpmf(y[i] | pred1 == 1 ? 1 - error_prob : error_prob);
    lp[2] = bernoulli_lpmf(y[i] | pred2 == 1 ? 1 - error_prob : error_prob);
    lp[3] = bernoulli_lpmf(y[i] | pred3 == 1 ? 1 - error_prob : error_prob);
    lp[4] = bernoulli_lpmf(y[i] | pred4 == 1 ? 1 - error_prob : error_prob);
    lp[5] = bernoulli_lpmf(y[i] | pred5 == 1 ? 1 - error_prob : error_prob);
    lp_marginal[i] = log_sum_exp(lp) - log(R);
  }
}
model {
  target += normal_lpdf(logit_error_prob | prior_logit_error_mean, prior_logit_error_sd);
  for (i in 1:ntrials) target += lp_marginal[i];
}
generated quantities {
  array[ntrials] real log_lik;
  for (i in 1:ntrials) log_lik[i] = lp_marginal[i];
  real lprior = normal_lpdf(logit_error_prob | prior_logit_error_mean, prior_logit_error_sd);
}

