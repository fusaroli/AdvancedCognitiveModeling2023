
// The input (data) for the model. n of trials and h for (right and left) hand
data {
  int<lower=1> n;
  array[n] int h;
  array[n] int other;
}

// The parameters accepted by the model. 
parameters {
  real bias; // how likely is the agent to pick right when the previous rate has no information (50-50)?
  real beta; // how strongly is previous rate impacting the decision?
  real<lower=0, upper=1> forgetting;
}

// The model to be estimated. 
model {
  
  vector[n] memory;
  // Priors
  target += beta_lpdf(forgetting | 1, 1);
  target += normal_lpdf(bias | 0, .3);
  target += normal_lpdf(beta | 0, .5);
  
  // Model, looping to keep track of memory
  for (trial in 1:n) {
    if (trial == 1) {
      memory[trial] = 0.5;
    }
    target += bernoulli_logit_lpmf(h[trial] | bias + beta * logit(memory[trial]));
    if (trial < n){
      memory[trial + 1] = (1 - forgetting) * memory[trial] + forgetting * other[trial];
      if (memory[trial + 1] == 0){memory[trial + 1] = 0.01;}
      if (memory[trial + 1] == 1){memory[trial + 1] = 0.99;}
    }
    
  }
}

