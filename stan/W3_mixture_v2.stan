//
// This Stan model defines a mixture of bernoulli (random bias + noise)
//

// The input (data) for the model. n of trials and h of heads
data {
 int<lower=1> n;
 array[n] int h;
 
 real noiseM; // prior mean for p of noise
 real<lower=0> noiseSd; // prior sd for p of noise
}

// The parameters accepted by the model. 
parameters {
  real bias;
  real noise; // p of noise
}

// The model to be estimated. 
model {
  target += normal_lpdf(bias | 0, 1);
  target += normal_lpdf(noise | noiseM, noiseSd);
  
  target += log_sum_exp(
            log(inv_logit(noise)) +  // p of noise
                    bernoulli_logit_lpmf(h | 0), // times post likelihood of the noise model
            log1m(inv_logit(noise)) + // 1 - p of noise
                    bernoulli_logit_lpmf(h | bias)); // times post likelihood of the bias model
}

generated quantities{
  real<lower=0, upper=1> noise_p;
  real<lower=0, upper=1> bias_p;
  noise_p = inv_logit(noise);
  bias_p = inv_logit(bias);
}
