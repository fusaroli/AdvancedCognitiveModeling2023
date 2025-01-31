
// This model infers a random bias from a sequences of 1s and 0s (right and left hand choices)
// The input (data) for the model. n of trials and the sequence of choices (right as 1, left as 0)
data {
 int<lower=1> n; // n of trials
 array[n] int h; // sequence of choices (right as 1, left as 0) as long as n
}
// The parameters that the model needs to estimate (theta)
parameters {
  real<lower=0, upper=1> theta; // rate or theta is a probability and therefore bound between 0 and 1 
}
// The model to be estimated (a bernoulli, parameter theta, prior on the theta)
model {
  // The prior for theta is a beta distribution alpha of 1, beta of 1, equivalent to a uniform between 0 and 1 
  target += beta_lpdf(theta | 1, 1);
  // N.B. you could also define the parameters of the priors as variables to be found in the data
  // target += beta_lpdf(theta | beta_alpha, beta_beta); BUT remember to add beta_alpha and beta_beta to the data list
  // The model consists of a bernoulli distribution (binomial w 1 trial only) with a rate theta
  target += bernoulli_lpmf(h | theta);
}

