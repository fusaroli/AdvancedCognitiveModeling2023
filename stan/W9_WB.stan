

data {
  int<lower=0> N;
  array[N] int y;
  vector[N] Source1;
  vector[N] Source2;
}

parameters {
  real weight1;
  real weight2;
}

model {
  target += normal_lpdf(weight1 | 1,1);
  target += normal_lpdf(weight2 | 1,1);
  for (n in 1:N)
    target += bernoulli_logit_lpmf(y[n] | weight1 * Source1[n] + weight2 * Source2[n]);
}

generated quantities{
  array[N] real log_lik;
  real w1;
  real w2;
  real w1_prior;
  real w2_prior;
  
  w1_prior = (normal_rng(1,1) - 0.5)*2 ;
  w2_prior = (normal_rng(1,1) - 0.5)*2 ;
  w1 = (weight1 - 0.5)*2;
  w2 = (weight2 - 0.5)*2;
  for (n in 1:N)
    log_lik[n] = bernoulli_logit_lpmf(y[n] | weight1 * Source1[n] + weight2 * Source2[n]);
  
}


