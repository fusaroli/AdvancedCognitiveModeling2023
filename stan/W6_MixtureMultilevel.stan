
//
// This Stan model defines a mixture of bernoulli (random bias + noise)
//
functions{
  real normal_lb_rng(real mu, real sigma, real lb) {
    real p = normal_cdf(lb | mu, sigma);  // cdf for bounds
    real u = uniform_rng(p, 1);
    return (sigma * inv_Phi(u)) + mu;  // inverse cdf for value
  }
}

// The input (data) for the model. n of trials and h of heads
data {
 int<lower = 1> trials;
 int<lower = 1> agents;
 array[trials, agents] int h;
}

// The parameters accepted by the model. 
parameters {
  real thetaM;
  real noiseM; // p of noise
  
  vector<lower = 0>[2] tau;
  matrix[2, agents] z_IDs;
  cholesky_factor_corr[2] L_u;
}

transformed parameters {
  matrix[agents,2] IDs;
  IDs = (diag_pre_multiply(tau, L_u) * z_IDs)';
 }

// The model to be estimated. 
model {
  target += normal_lpdf(thetaM | 0, 1);
  target += normal_lpdf(tau[1] | 0, .3)  -
    normal_lccdf(0 | 0, .3);

  target += normal_lpdf(noiseM | -1, .5);
  target += normal_lpdf(tau[2] | 0, .3)  -
    normal_lccdf(0 | 0, .3);
    
  target += lkj_corr_cholesky_lpdf(L_u | 2);
  
  target += std_normal_lpdf(to_vector(z_IDs));

  for (i in 1:agents)
    target += log_sum_exp(
            log(inv_logit(noiseM + IDs[i,2])) +  // p of noise
                    bernoulli_logit_lpmf(h[,i] | 0), // times post likelihood of the noise model
            log1m(inv_logit(noiseM + IDs[i,2])) + // 1 - p of noise
                    bernoulli_logit_lpmf(h[,i] | thetaM + IDs[i,1])); // times post likelihood of the bias model
                    

}

generated quantities{
  real thetaM_prior;
  real<lower=0> thetaSD_prior;
  real noiseM_prior;
  real<lower=0> noiseSD_prior;
  real<lower=0, upper=1> theta_prior;
  real<lower=0, upper=1> noise_prior;
  real<lower=0, upper=1> theta_posterior;
  real<lower=0, upper=1> noise_posterior;
  
  array[trials,agents] int<lower=0, upper = trials> prior_noise;
  array[trials,agents] int<lower=0, upper = trials> posterior_noise;
  array[trials,agents] int<lower=0, upper = trials> prior_preds;
  array[trials,agents] int<lower=0, upper = trials> posterior_preds;

  array[trials, agents] real log_lik;

  thetaM_prior = normal_rng(0,1);
  thetaSD_prior = normal_lb_rng(0,0.3,0);
  theta_prior = inv_logit(normal_rng(thetaM_prior, thetaSD_prior));
  noiseM_prior = normal_rng(-1,.5);
  noiseSD_prior = normal_lb_rng(0,0.3,0);
  noise_prior = inv_logit(normal_rng(noiseM_prior, noiseSD_prior));
  
  theta_posterior = inv_logit(normal_rng(thetaM, tau[1]));
  noise_posterior = inv_logit(normal_rng(noiseM, tau[2]));
  
   
   for (i in 1:agents){
     
    for (t in 1:trials){
      
      prior_noise[t,i] = bernoulli_rng(noise_prior);
      posterior_noise[t,i] = bernoulli_rng(inv_logit(noiseM + IDs[i,2]));
      
      if(prior_noise[t,i]==1){
        prior_preds[t,i] = bernoulli_rng(theta_prior);
      } else{
        prior_preds[t,i] = bernoulli_rng(0.5);
      }
      if(posterior_noise[t,i]==1){
        posterior_preds[t,i] = bernoulli_rng(inv_logit(thetaM + IDs[i,1]));
      } else{
        posterior_preds[t,i] = bernoulli_rng(0.5);
      }
      
      
      log_lik[t,i] = log_sum_exp(
            log(inv_logit(noiseM + IDs[i,2])) +  // p of noise
                    bernoulli_logit_lpmf(h[t,i] | 0), // times post likelihood of the noise model
            log1m(inv_logit(noiseM + IDs[i,2])) + // 1 - p of noise
                    bernoulli_logit_lpmf(h[t,i] | thetaM + IDs[i,1])); // times post likelihood of the bias model
      
    }
  }
  
}

