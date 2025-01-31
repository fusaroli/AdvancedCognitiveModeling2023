

data {
  int<lower=0> N;
  array[N] int y;
  array[N] int Source1;
  array[N] int Source2;
  array[N] int Truth;
}

parameters {
  real bias;
  real w1;
  real learningRate;
  // Estimated rates
  array[N] real theta1;
  array[N] real theta2;
}

model {
  array[N] real w2;
  // Priors
  target +=  normal_lpdf(bias | 0, 1);
  target +=  normal_lpdf(w1 | 0, 1);
  //target +=  normal_lpdf(w2[1] | 0, 1);
  // Estimating rates
  target +=  binomial_logit_lpmf(Source1 | 8, theta1); 
  target +=  binomial_logit_lpmf(Source2 | 4, theta2); 
  
  w2[1] = 1;
  for (n in 1:N){
    target +=  bernoulli_logit_lpmf(y[n] | bias + w1 * theta1[n] + w2[n] * theta2[n]);
    
    if (n < N){
      if (Truth[n] == Source2[n]){
        w2[n + 1] = w2[n] + learningRate * (1 - w2[n]);  // Assuming max(w1) = 1
      }
      if (Truth[n] != Source2[n]){
        w2[n + 1] = w2[n] + learningRate * (0 - w2[n]);  // Assuming max(w1) = 1
      }
    }
  }
  
}

