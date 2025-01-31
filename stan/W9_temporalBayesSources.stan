

data {
  int<lower=0> N;
  array[N] int y;
  array[N] int Source1; // how many red
  array[N] int marbles1; // how many sampled at that
}

parameters {
  real bias;
  real w1; 
  real w2;  
  array[N] real theta1; // Estimated rate
}

transformed parameters {
  array[N] real theta2;
  theta2[1] = 0;
  for (n in 2:N){
    theta2[n] = bias + w1 * theta1[n] + w2 * theta2[n-1];
    }
}

model {
  target += normal_lpdf(bias | 0, 1);
  target += normal_lpdf(w1 | 0, 1);
  target += normal_lpdf(w2 | 0, 1);
  target += binomial_logit_lpmf(Source1 | marbles1, theta1); 
  target += bernoulli_logit_lpmf(y | bias + w1 * to_vector(theta1) + w2 * to_vector(theta2));
}

generated quantities{
  array[N] real log_lik;
  real bias_prior;
  real w1_prior;
  real w2_prior;
  bias_prior = normal_rng(0, 1) ;
  w1_prior = normal_rng(0, 1) ;
  w2_prior = normal_rng(0, 1) ;
  for (n in 1:N)
    log_lik[n]= bernoulli_logit_lpmf(y[n] | bias + w1 * theta1[n] + w2 * theta2[n]);
}



