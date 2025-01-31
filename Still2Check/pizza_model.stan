
data {
  int<lower=0> N;                   // Number of observations
  vector[N] time;                   // Time points
  vector[N] temp;                   // Observed temperatures
  int<lower=0> n_raters;           // Number of raters
  array[N] int<lower=1,upper=n_raters> rater;  // Rater indices
  vector[n_raters] Ti;             // Initial temperatures
  real Tinf;                       // Flame temperature
}

parameters {
  vector<lower=0>[n_raters] HOT;   // Heating coefficients
  vector<lower=0>[n_raters] sigma; // Measurement error
}

model {
  vector[N] mu;
  
  // Physics-based temperature prediction
  for (i in 1:N) {
    mu[i] = Tinf + (Ti[rater[i]] - Tinf) * exp(-HOT[rater[i]] * time[i]);
  }
  
  // Prior distributions
  target += normal_lpdf(HOT | 0.005, 0.005);    // Prior for heating rate
  target += exponential_lpdf(sigma | 1);         // Prior for measurement error
  
  // Likelihood
  target += normal_lpdf(temp | mu, sigma[rater]);
}

