
functions{
  // Helper function for generating truncated normal random numbers
  real normal_lb_rng(real mu, real sigma, real lb) {
    real p = normal_cdf(lb | mu, sigma);  // cdf for bounds
    real u = uniform_rng(p, 1);
    return (sigma * inv_Phi(u)) + mu;  // inverse cdf for value
  }
}

data {
 int<lower = 1> trials;        // Number of trials
 array[trials] int h;          // Choices (0/1)
 vector[trials] win;           // Win signal for each trial
 vector[trials] lose;          // Lose signal for each trial
}

parameters {
  real alpha;                  // Baseline bias parameter
  real winB;                   // Win-stay parameter
  real loseB;                  // Lose-shift parameter
}

model {
  // Priors
  target += normal_lpdf(alpha | 0, .3);   // Prior for baseline bias
  target += normal_lpdf(winB | 1, 1);     // Prior for win-stay parameter
  target += normal_lpdf(loseB | 1, 1);    // Prior for lose-shift parameter
  
  // Likelihood: WSLS choice model
  // Remember that in the first trial we ensured win and lose have a value of 0,
  // which correspond to a fixed 0.5 probability (0 on logit scale)
  // since there is no previous outcome to guide the choice
  target += bernoulli_logit_lpmf(h | alpha + winB * win + loseB * lose);

}

generated quantities{
   // Prior predictive samples
   real alpha_prior;
   real winB_prior;
   real loseB_prior;
   
   // Posterior and prior predictions
   array[trials] int prior_preds;
   array[trials] int posterior_preds;
   
   // Log likelihood for model comparison
   vector[trials] log_lik;

   // Generate prior samples
   alpha_prior = normal_rng(0, 1);
   winB_prior = normal_rng(0, 1);
   loseB_prior = normal_rng(0, 1);
   
   // Prior predictive simulations
   for (t in 1:trials) {
     prior_preds[t] = bernoulli_logit_rng(alpha_prior + winB_prior * win[t] + loseB_prior * lose[t]);
   }
   
   // Posterior predictive simulations
   for (t in 1:trials) {
     posterior_preds[t] = bernoulli_logit_rng(alpha + winB * win[t] + loseB * lose[t]);
   }
   
   // Calculate log likelihood for each observation
   for (t in 1:trials){
     log_lik[t] = bernoulli_logit_lpmf(h[t] | alpha + winB * win[t] + loseB * lose[t]);
   }
}

