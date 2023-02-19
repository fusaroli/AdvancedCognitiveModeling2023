//
// This Stan model infers a random bias from a sequences of 1s and 0s (right and left hand choices)
//

// The input (data) for the model. n of trials and sequence of 0s and 1s
data {
 int<lower=1> n;
 array[n] int h;
}

// The parameters accepted by the model. 
parameters {
  real theta; // note it is unbounded as we now work on log odds
}

// The model to be estimated. 
model {
  target += normal_lpdf(theta | 0, 1);
  
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

