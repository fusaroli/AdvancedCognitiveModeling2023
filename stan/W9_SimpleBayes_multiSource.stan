
data {
  int<lower=0> N;              // number of observations
  int<lower=1> K;              // number of predictors
  array[N, K] real X;              // predictor matrix
  array[N] int<lower=0,upper=1> y;   // outcome vector
}
parameters {
  real bias;                  // intercept
}
model {
  // prior
  bias ~ normal(0, 1);
  
  // likelihood
  target += bernoulli_logit_lpmf(y | columns_dot_product(X, rep_vector(bias, N)));
}

generated quantities {
  // log likelihood for model comparison
  real log_lik = 0;
  for (n in 1:N) {
    log_lik += bernoulli_logit_lpmf(y[n] | bias + sum(X[n]));
  }
}
