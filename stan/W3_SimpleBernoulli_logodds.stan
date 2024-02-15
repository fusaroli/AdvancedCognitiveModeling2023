
// This model infers a random bias from a sequences of 1s and 0s (right and left hand choices)

// The input (data) for the model. n of trials and the sequence of choices (right as 1, left as 0)
data {
 int<lower=1> n; // n of trials
 array[n] int h; // sequence of choices (right as 1, left as 0) as long as n
}

// The parameters that the model needs to estimate (theta)
parameters {
    real theta; // note it is unbounded as we now work on log odds
}

// The model to be estimated (a bernoulli, parameter theta, prior on the theta)
model {
  // The prior for theta on a log odds scale is a normal distribution with a mean of 0 and a sd of 1.
  // This covers most of the probability space between 0 and 1, after being converted to probability.
  target += normal_lpdf(theta | 0, 1);
  // as before the parameters of the prior could be fed as variables
  // target += normal_lpdf(theta | normal_mu, normal_sigma);
  
  // The model consists of a bernoulli distribution (binomial w 1 trial only) with a rate theta,
  // note we specify it uses a logit link (theta is in logodds)
  target += bernoulli_logit_lpmf(h | theta);
  
}

