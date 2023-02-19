//
// This Stan model infers a random bias from a sequences of 1s and 0s (right and left hand choices)
//

// The input (data) for the model. n of trials and the sequence of choices
data {
 int<lower=1> n;
 array[n] int h;
}

// The parameters accepted by the model. 
parameters {
  real<lower=0, upper=1> theta; // rate or theta is a probability and therefore bound between 0 and 1 
}

// The model to be estimated. 
model {
  // The prior for theta is a uniform distribution between 0 and 1
  target += beta_lpdf(theta | 1, 1);
  
  // The model consists of a bernoulli distribution (binomial w 1 trial only) with a rate theta
  target += bernoulli_lpmf(h | theta);
}

