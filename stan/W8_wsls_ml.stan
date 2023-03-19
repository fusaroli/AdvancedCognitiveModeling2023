
functions{
  real normal_lb_rng(real mu, real sigma, real lb) {
    real p = normal_cdf(lb | mu, sigma);  // cdf for bounds
    real u = uniform_rng(p, 1);
    return (sigma * inv_Phi(u)) + mu;  // inverse cdf for value
  }
}

// The input (data) for the model. 
data {
 int<lower = 1> trials;
 int<lower = 1> agents;
 array[trials, agents] int h;
 array[trials, agents] real win;
 array[trials, agents] real lose;
}

parameters {
  real winM;
  real loseM;
  vector<lower = 0>[2] tau;
  matrix[2, agents] z_IDs;
  cholesky_factor_corr[2] L_u;
}

transformed parameters {
  matrix[agents,2] IDs;
  IDs = (diag_pre_multiply(tau, L_u) * z_IDs)';
 }

model {
  target += normal_lpdf(winM | 0, 1);
  target += normal_lpdf(tau[1] | 0, .3)  -
    normal_lccdf(0 | 0, .3);
  target += normal_lpdf(loseM | 0, .3);
  target += normal_lpdf(tau[2] | 0, .3)  -
    normal_lccdf(0 | 0, .3);
  target += lkj_corr_cholesky_lpdf(L_u | 2);

  target += std_normal_lpdf(to_vector(z_IDs));
  
  for (i in 1:agents)
    target += bernoulli_logit_lpmf(h[,i] | to_vector(win[,i]) * (winM + IDs[i,1]) +  to_vector(lose[,i]) * (loseM + IDs[i,2]));
}

generated quantities{
   real winM_prior;
   real<lower=0> winSD_prior;
   real loseM_prior;
   real<lower=0> loseSD_prior;
   
   real win_prior;
   real lose_prior;
   
   array[trials,agents] int<lower=0, upper = trials> prior_preds;
   array[trials,agents] int<lower=0, upper = trials> posterior_preds;
   
   array[trials, agents] real log_lik;


   winM_prior = normal_rng(0,1);
   winSD_prior = normal_lb_rng(0,0.3,0);
   loseM_prior = normal_rng(0,1);
   loseSD_prior = normal_lb_rng(0,0.3,0);
   
   win_prior = normal_rng(winM_prior, winSD_prior);
   lose_prior = normal_rng(loseM_prior, loseSD_prior);
   
   for (i in 1:agents){
      prior_preds[,i] = binomial_rng(trials, inv_logit(to_vector(win[,i]) * (win_prior) +  to_vector(lose[,i]) * (lose_prior)));
      posterior_preds[,i] = binomial_rng(trials, inv_logit(to_vector(win[,i]) * (winM + IDs[i,1]) +  to_vector(lose[,i]) * (loseM + IDs[i,2])));
      
    for (t in 1:trials){
      log_lik[t,i] = bernoulli_logit_lpmf(h[t,i] | to_vector(win[,i]) * (winM + IDs[i,1]) +  to_vector(lose[,i]) * (loseM + IDs[i,2]));
    }
  }
  
}



