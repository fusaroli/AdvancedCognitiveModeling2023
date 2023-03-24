

functions{
  real weight_f(real L_raw, real w_raw) {
    real L;
    real w;
    L = exp(L_raw);
    w = 0.5 + inv_logit(w_raw)/2;
    return log((w * L + 1 - w)./((1 - w) * L + w));
  }
}


data {
  int<lower=0> N;
  array[N] int y;
  vector[N] Source1;
  vector[N] Source2;
}

parameters {
  real bias;
  real weight1;
  real weight2;
}

model {
  target += normal_lpdf(bias | 0, 1);
  target += normal_lpdf(weight1 | 0, 1.5);
  target += normal_lpdf(weight2 | 0, 1.5);
  
  for (n in 1:N){  
  target += bernoulli_logit_lpmf(y[n] | bias + weight_f(Source1[n], weight1) + weight_f(Source2[n], weight2));
  }
}

generated quantities{
  array[N] real log_lik;
  real bias_prior;
  real w1_prior;
  real w2_prior;
  real w1;
  real w2;
  
  bias_prior = normal_rng(0,1);
  w1_prior = normal_rng(0,1.5);
  w2_prior = normal_rng(0,1.5);
  
  w1_prior = 0.5 + inv_logit(normal_rng(0,1))/2;
  w2_prior = 0.5 + inv_logit(normal_rng(0,1))/2;
  w1 = 0.5 + inv_logit(weight1)/2;
  w2 = 0.5 + inv_logit(weight2)/2;
  
  for (n in 1:N){  
    log_lik[n] = bernoulli_logit_lpmf(y[n] | bias + weight_f(Source1[n], weight1) +
      weight_f(Source2[n], weight2));
  }
  
}


