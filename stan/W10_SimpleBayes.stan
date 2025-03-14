
data {
  int<lower=0> N;
  array[N] int y;
  array[N] real<lower=0, upper = 1> Source1;
  array[N] real<lower=0, upper = 1> Source2;
}

transformed data{
  array[N] real l_Source1;
  array[N] real l_Source2;
  l_Source1 = logit(Source1);
  l_Source2 = logit(Source2);
}

parameters {
  real bias;
}

model {
  target +=  normal_lpdf(bias | 0, 1);
  target +=  bernoulli_logit_lpmf(y | bias + to_vector(l_Source1) + to_vector(l_Source2));
}

generated quantities{
  real bias_prior;
  array[N] real log_lik;
  
  bias_prior = normal_rng(0, 1);
  
  for (n in 1:N){  
    log_lik[n] = bernoulli_logit_lpmf(y[n] | bias + l_Source1[n] +  l_Source2[n]);
  }
}

