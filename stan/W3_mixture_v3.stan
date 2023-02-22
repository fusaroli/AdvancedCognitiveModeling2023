//
// This STAN model infers a random bias from a sequences of 1s and 0s (heads and tails)
//

// The input (data) for the model. n of trials and h of heads
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
  // The prior for theta is a uniform distribution between 0 and 1
  target += normal_lpdf(theta | prior_mean, prior_sd);
  
  // The model consists of a binomial distributions with a rate theta
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

