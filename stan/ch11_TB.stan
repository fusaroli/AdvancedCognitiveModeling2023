
data {
  int<lower=0> N;
  array[N] int y;
  array[N] real <lower = 0, upper = 1> Source1; 
}

transformed data {
  array[N] real l_Source1;
  l_Source1 = logit(Source1);
}

parameters {
  real bias;
  // meaningful weights are btw 0.5 and 1 (theory reasons)
  real<lower = 0.5, upper = 1> w1; 
  real<lower = 0.5, upper = 1> w2;
}

transformed parameters {
  real<lower = 0, upper = 1> weight1;
  real<lower = 0, upper = 1> weight2;
  array[N] real l_Source2;

  // weight parameters are rescaled to be on a 0-1 scale (0 -> no effects; 1 -> face value)
  weight1 = (w1 - 0.5) * 2;  
  weight2 = (w2 - 0.5) * 2;
  
  // Initialize first time step with neutral prior (logit(0.5) = 0)
  l_Source2[1] = 0;
  
  // Calculate belief propagation over time
  for (n in 2:N){
    l_Source2[n] = bias + weight1 * l_Source1[n] + weight2 * l_Source2[n-1];
  }
}

model {
  // Priors
  target += normal_lpdf(bias | 0, 1);
  target += beta_lpdf(weight1 | 1, 1);
  target += beta_lpdf(weight2 | 1, 1);
  
  // First time step has only new evidence, no prior
  target += bernoulli_logit_lpmf(y[1] | bias + weight1 * l_Source1[1] + weight2 * l_Source2[1]);
  
  // Subsequent time steps combine new evidence with prior belief
  for (n in 2:N){  
    target += bernoulli_logit_lpmf(y[n] | l_Source2[n]);
  }
}

generated quantities{
  array[N] real log_lik;
  real bias_prior;
  real w1_prior;
  real w2_prior;
  bias_prior = normal_rng(0, 1) ;
  w1_prior = 0.5 + inv_logit(normal_rng(0, 1))/2 ;
  w2_prior = 0.5 + inv_logit(normal_rng(0, 1))/2 ;
  for (n in 1:N)
    log_lik[n]= bernoulli_logit_lpmf(y[n] | l_Source2[n]);
}

