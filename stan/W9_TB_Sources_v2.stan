data {
  int<lower=0> N;
  array[N] int ColorChoice;
  array[N] int SamplingChoice;
  array[N] int newBlock;
  array[N] int Source1; // how many red
  array[N] int marbles1; // how many sampled at that
}

parameters {
  real w1; 
  real w2;  
  array[N] real theta1; // Estimated rate
}

transformed parameters {
  array[N] real theta2;
  for (n in 1:N){
    if (newBlock[n] == 1) {
      theta2[n] = 0;
    } else {
      theta2[n] = w1 * theta1[n] + w2 * theta2[n-1];
    }
  }
}

model {
  target += normal_lpdf(w1 | 0, 1);
  target += normal_lpdf(w2 | 0, 1);
  target += binomial_logit_lpmf(Source1 | marbles1, theta1); 
  target += bernoulli_logit_lpmf(SamplingChoice | abs(w1 * to_vector(theta1) + w2 * to_vector(theta2)));
  for (n in 1:N){
    if (SamplingChoice[n] == 1){
      target += bernoulli_logit_lpmf(ColorChoice[n] | w1 * theta1[n] + w2 * theta2[n]);
    }
  }
}