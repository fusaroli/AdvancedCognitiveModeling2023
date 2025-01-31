data {
  int<lower=0> N;
  array[N] int y;
  array[N] int Source1;
  array[N] int Source2;
  array[N] int Truth;
}

parameters {
  real w1;
  real learningRate;
  array[N] real w2;
  // Estimated rates
  array[N] real theta1;
  array[N] real theta2;
}

transformed parameters {
  real learningRate;
  learningRate = inv_logit(l_learningRate);
}

model {
  // Priors
  target +=  beta_lpdf(w1 | 0, 1);
  target +=  beta_lpdf(w2[1] | 0, 1);
  target +=  normal_lpdf(l_learningRate | 0, 1);
  // Estimating rates
  target +=  binomial_logit_lpmf(Source1 | 8, theta1); 
  target +=  binomial_logit_lpmf(Source2 | 4, theta2); 
  
  for (n in 1:N){
    target +=  bernoulli_logit_lpmf(y[n] | w1 * to_vector(theta1[n]) + w2[n] * to_vector(theta2[n]));
    
    if (n < N){
      if (Truth[n] == Source2[n]){
        w2[n + 1] = w2[n] + learningRate * (1 - w2);  // Assuming max(w2) = 1
      }
      if (Truth[n] != Source2[n]){
        w2[n + 1] = w2[n] + learningRate * (0 - w2);  // Assuming max(w2) = 1
      }
    }
  }
}
