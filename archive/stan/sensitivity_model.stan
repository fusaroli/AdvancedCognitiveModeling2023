
  functions {
    real normal_lb_rng(real mu, real sigma, real lb) {
      real p = normal_cdf(lb | mu, sigma);
      real u = uniform_rng(p, 1);
      return (sigma * inv_Phi(u)) + mu;
    }
  }
  data {
    int<lower=1> trials;
    int<lower=1> agents;
    array[trials, agents] int<lower=0, upper=1> h;
    real prior_mean_theta;
    real<lower=0> prior_sd_theta;
    real<lower=0> prior_scale_theta_sd;
  }
  parameters {
    real thetaM;
    real<lower=0> thetaSD;
    array[agents] real theta;
  }
  model {
    // Population-level priors with specified hyperparameters
    target += normal_lpdf(thetaM | prior_mean_theta, prior_sd_theta);
    target += normal_lpdf(thetaSD | 0, prior_scale_theta_sd) - normal_lccdf(0 | 0, prior_scale_theta_sd);
    // Agent-level model
    target += normal_lpdf(theta | thetaM, thetaSD);
    // Likelihood
    for (i in 1:agents) {
      target += bernoulli_logit_lpmf(h[,i] | theta[i]);
    }
  }
  generated quantities {
    real thetaM_prob = inv_logit(thetaM);
  }
  
