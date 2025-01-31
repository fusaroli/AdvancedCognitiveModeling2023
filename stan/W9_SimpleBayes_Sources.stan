data {
  int<lower=0> N;
  array[N] int y;
  array[N] int Source1;
  array[N] int Source2;
}

parameters {
  real bias;
  real w1;
  real w2;
  // Estimated rates
  array[N] real theta1;
  array[N] real theta2;
}

model {
  // Priors
  target +=  normal_lpdf(bias | 0, 1);
  target +=  normal_lpdf(w1 | 0, 1);
  target +=  normal_lpdf(w2 | 0, 1);
  // Estimating rates
  target +=  binomial_logit_lpmf(Source1 | 8, theta1); 
  target +=  binomial_logit_lpmf(Source2 | 4, theta2); 
  target +=  bernoulli_logit_lpmf(y | bias + w1 * to_vector(theta1) + w2 * to_vector(theta2));
  
}