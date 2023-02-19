//
// This STAN model infers a random bias from a sequences of 1s and 0s (heads and tails)
//

// The input (data) for the model. n of trials and h of heads
data {
 int<lower=1> n;
 array[n] int h;
 vector<lower=0, upper=1>[n] memory;
}

// The parameters accepted by the model. 
parameters {
  real alpha;
  real beta;
}



// The model to be estimated. 
model {
  // The prior for theta is a uniform distribution between 0 and 1
  target += normal_lpdf(alpha | 0, 1);
  target += normal_lpdf(beta | 0, .3);
  
  // The model consists of a binomial distributions with a rate theta
  target += bernoulli_logit_lpmf(h | alpha + beta * memory);
}

generated quantities{
  array[n] int preds;
  real alpha_prior;
  real beta_prior;

  alpha_prior = normal_rng(0, 1);
  beta_prior = normal_rng(0, 0.3);
  preds = binomial_rng(n, inv_logit(alpha + beta * memory));
}

