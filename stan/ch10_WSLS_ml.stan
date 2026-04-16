
functions{
  real normal_lb_rng(real mu, real sigma, real lb) {
    real p = normal_cdf(lb | mu, sigma);  // cdf for bounds
    real u = uniform_rng(p, 1);
    return (sigma * inv_Phi(u)) + mu;  // inverse cdf for value
  }
}

// The input (data) for the model. 
data {
 int<lower = 1> trials;           // Number of trials
 int<lower = 1> agents;           // Number of agents
 array[trials, agents] int h;      // Choice data (0/1)
 array[trials, agents] real win;   // Win signals
 array[trials, agents] real lose;  // Lose signals
}

parameters {
  // Population-level parameters
  real winM;                    // Population mean for win-stay parameter
  real loseM;                   // Population mean for lose-shift parameter
  
  // Population standard deviations
  vector<lower = 0>[2] tau;     // SDs for [win, lose] parameters
  
  // Individual z-scores (non-centered parameterization)
  matrix[2, agents] z_IDs;
  
  // Correlation matrix
  cholesky_factor_corr[2] L_u;
}

transformed parameters {
  // Individual parameters (constructed from non-centered parameterization)
  matrix[agents, 2] IDs;
  IDs = (diag_pre_multiply(tau, L_u) * z_IDs)';
}

model {
  // Population-level priors
  target += normal_lpdf(winM | 0, 1);     // Prior for win-stay mean
  target += normal_lpdf(tau[1] | 0, .3) - normal_lccdf(0 | 0, .3);  // Half-normal for SD
  
  target += normal_lpdf(loseM | 0, .3);   // Prior for lose-shift mean
  target += normal_lpdf(tau[2] | 0, .3) - normal_lccdf(0 | 0, .3);  // Half-normal for SD
  
  // Prior for correlation matrix
  target += lkj_corr_cholesky_lpdf(L_u | 2);
  
  // Prior for individual z-scores
  target += std_normal_lpdf(to_vector(z_IDs));
  
  // Likelihood
  for (i in 1:agents)
    target += bernoulli_logit_lpmf(h[,i] | to_vector(win[,i]) * (winM + IDs[i,1]) + 
                                            to_vector(lose[,i]) * (loseM + IDs[i,2]));
}

generated quantities{
   // Prior predictive samples
   real winM_prior;
   real<lower=0> winSD_prior;
   real loseM_prior;
   real<lower=0> loseSD_prior;
   
   real win_prior;
   real lose_prior;
   
   // Posterior predictive samples for various scenarios
   array[trials, agents] int prior_preds;
   array[trials, agents] int posterior_preds;
   
   // Log likelihood for model comparison
   array[trials, agents] real log_lik;

   // Generate prior samples
   winM_prior = normal_rng(0, 1);
   winSD_prior = normal_lb_rng(0, 0.3, 0);
   loseM_prior = normal_rng(0, 1);
   loseSD_prior = normal_lb_rng(0, 0.3, 0);
   
   win_prior = normal_rng(winM_prior, winSD_prior);
   lose_prior = normal_rng(loseM_prior, loseSD_prior);
   
   // Generate predictions
   for (i in 1:agents){
      // Prior predictive simulations
      for (t in 1:trials) {
        prior_preds[t, i] = bernoulli_logit_rng(
          win[t, i] * win_prior + lose[t, i] * lose_prior
        );
      }
      
      // Posterior predictive simulations
      for (t in 1:trials) {
        posterior_preds[t, i] = bernoulli_logit_rng(
          win[t, i] * (winM + IDs[i, 1]) + lose[t, i] * (loseM + IDs[i, 2])
        );
      }
      
      // Calculate log likelihood for each observation
      for (t in 1:trials){
        log_lik[t, i] = bernoulli_logit_lpmf(
          h[t, i] | win[t, i] * (winM + IDs[i, 1]) + lose[t, i] * (loseM + IDs[i, 2])
        );
      }
   }
}

