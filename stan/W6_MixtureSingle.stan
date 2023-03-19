
// This Stan model defines a mixture of bernoulli (random bias + noise)
//

// The input (data) for the model. n of trials and h of heads
data {
 int<lower=1> n;
 array[n] int h;
}

// The parameters accepted by the model. 
parameters {
  real bias;
  real noise;
}

// The model to be estimated. 
model {
  // The prior for theta is a uniform distribution between 0 and 1
  target += normal_lpdf(bias | 0, 1);
  target += normal_lpdf(noise | 0, 1);
  
  // The model consists of a binomial distributions with a rate theta
  target += log_sum_exp(log(inv_logit(noise)) +
            bernoulli_logit_lpmf(h | 0),
            log1m(inv_logit(noise)) +  bernoulli_logit_lpmf(h | bias));
            
}
generated quantities{
  real<lower=0, upper=1> noise_p;
  real<lower=0, upper=1> bias_p;
  noise_p = inv_logit(noise);
  bias_p = inv_logit(bias);
}


