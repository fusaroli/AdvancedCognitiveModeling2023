
  // The input (data) for the model
data {
  int<lower=1> n;
  array[n] int h;
  real prior_mean;
  real<lower=0> prior_sd;
}

// The parameters accepted by the model. 
parameters {
  real theta;
}

// The model to be estimated. 
model {
  // Prior
  target += normal_lpdf(theta | prior_mean, prior_sd);
  
  // Model
  target += bernoulli_logit_lpmf(h | theta);
}

generated quantities{
  real<lower=0, upper=1> theta_prior;
  real<lower=0, upper=1> theta_posterior;
  int<lower=0, upper=n> prior_preds;
  int<lower=0, upper=n> posterior_preds;
  
  theta_prior = inv_logit(normal_rng(0,1));
  theta_posterior = inv_logit(theta);
  prior_preds = binomial_rng(n, theta_prior);
  posterior_preds = binomial_rng(n, inv_logit(theta));
  
}

