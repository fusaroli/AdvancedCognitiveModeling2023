
// Multilevel mixture model for binary choice data
// Allows individual differences in both bias and mixture weights

functions {
  real normal_lb_rng(real mu, real sigma, real lb) {
    real p = normal_cdf(lb | mu, sigma);  // cdf for bounds
    real u = uniform_rng(p, 1);
    return (sigma * inv_Phi(u)) + mu;  // inverse cdf for value
  }
}

data {
  int<lower=1> trials;       // Number of trials per agent
  int<lower=1> agents;       // Number of agents
  array[trials, agents] int h;  // Choice data (0/1)
}

parameters {
  // Population-level parameters
  real biasM;                // Population mean of bias (logit scale)
  real noiseM;               // Population mean of noise proportion (logit scale)
  
  // Population standard deviations
  vector<lower=0>[2] tau;    // SDs for [bias, noise]
  
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
  // Priors for population means
  target += normal_lpdf(biasM | 0, 1);     // Prior for population bias mean
  target += normal_lpdf(noiseM | -1, 0.5); // Prior for population noise mean (favoring lower noise)
  
  // Priors for population SDs (half-normal)
  target += normal_lpdf(tau[1] | 0, 0.3) - normal_lccdf(0 | 0, 0.3);
  target += normal_lpdf(tau[2] | 0, 0.3) - normal_lccdf(0 | 0, 0.3);
  
  // Prior for correlation matrix
  target += lkj_corr_cholesky_lpdf(L_u | 2);
  
  // Prior for individual z-scores
  target += std_normal_lpdf(to_vector(z_IDs));
  
  // Likelihood
  for (i in 1:agents) {
    target += log_sum_exp(
      log(inv_logit(noiseM + IDs[i,2])) +          // Prob of random process for agent i
      bernoulli_logit_lpmf(h[,i] | 0),             // Likelihood under random process
      
      log1m(inv_logit(noiseM + IDs[i,2])) +        // Prob of biased process for agent i
      bernoulli_logit_lpmf(h[,i] | biasM + IDs[i,1])  // Likelihood under biased process
    );
  }
}

generated quantities {
  // Prior predictive samples
  real biasM_prior = normal_rng(0, 1);
  real<lower=0> biasSD_prior = normal_lb_rng(0, 0.3, 0);
  real noiseM_prior = normal_rng(-1, 0.5);
  real<lower=0> noiseSD_prior = normal_lb_rng(0, 0.3, 0);
  
  // Transform to probability scale for easier interpretation
  real<lower=0, upper=1> bias_mean = inv_logit(biasM);
  real<lower=0, upper=1> noise_mean = inv_logit(noiseM);
  
  // Correlation between parameters
  corr_matrix[2] Omega = multiply_lower_tri_self_transpose(L_u);
  real bias_noise_corr = Omega[1,2];
  
  // For each agent, generate individual parameters and predictions
  array[agents] real<lower=0, upper=1> agent_bias;
  array[agents] real<lower=0, upper=1> agent_noise;
  array[trials, agents] int pred_component;
  array[trials, agents] int pred_choice;
  array[trials, agents] real log_lik;
  
  // Calculate individual parameters and generate predictions
  for (i in 1:agents) {
    // Individual parameters
    agent_bias[i] = inv_logit(biasM + IDs[i,1]);
    agent_noise[i] = inv_logit(noiseM + IDs[i,2]);
    
    // Log likelihood calculations
    for (t in 1:trials) {
      log_lik[t,i] = log_sum_exp(
        log(agent_noise[i]) + bernoulli_logit_lpmf(h[t,i] | 0),
        log1m(agent_noise[i]) + bernoulli_logit_lpmf(h[t,i] | logit(agent_bias[i]))
      );
    }
    
    // Generate predictions
    for (t in 1:trials) {
      pred_component[t,i] = bernoulli_rng(agent_noise[i]);
      if (pred_component[t,i] == 1) {
        // Random component
        pred_choice[t,i] = bernoulli_rng(0.5);
      } else {
        // Biased component
        pred_choice[t,i] = bernoulli_rng(agent_bias[i]);
      }
    }
  }
}

