
// The input (data) for the model. n of trials and h for (right and left) hand
data {
 int<lower=1> n;
 array[n] int h;
 vector[n] memory; // here we add the new variable between 0.01 and .99
}

// The parameters accepted by the model. 
parameters {
  real bias; // how likely is the agent to pick right when the previous rate has no information (50-50)?
  real beta; // how strongly is previous rate impacting the decision?
}

// The model to be estimated. 
model {
  // priors
  target += normal_lpdf(bias | 0, .3);
  target += normal_lpdf(beta | 0, .5);
  
  // model
  target += bernoulli_logit_lpmf(h | bias + beta * logit(memory));
}

