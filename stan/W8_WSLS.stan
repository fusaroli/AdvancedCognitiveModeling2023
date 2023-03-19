

functions{
  real normal_lb_rng(real mu, real sigma, real lb) {
    real p = normal_cdf(lb | mu, sigma);  // cdf for bounds
    real u = uniform_rng(p, 1);
    return (sigma * inv_Phi(u)) + mu;  // inverse cdf for value
  }
}

data {
 int<lower = 1> trials;
 array[trials] int h;
 vector[trials] win;
 vector[trials] lose;
}

parameters {
  real alpha;
  real winB;
  real loseB;
}

model {
  target += normal_lpdf(alpha | 0, .3);
  target += normal_lpdf(winB | 1, 1);
  target += normal_lpdf(loseB | 1, 1);
  target += bernoulli_logit_lpmf(h | alpha + winB * win + loseB * lose);
}

generated quantities{
   real alpha_prior;
   real winB_prior;
   real loseB_prior;
   array[trials] int prior_preds;
   array[trials] int posterior_preds;
   vector[trials] log_lik;

   alpha_prior = normal_rng(0, 1);
   winB_prior = normal_rng(0, 1);
   loseB_prior = normal_rng(0, 1);
   
   prior_preds = bernoulli_rng(inv_logit(winB_prior * win +  loseB_prior * lose));
   posterior_preds = bernoulli_rng(inv_logit(winB * win +  loseB * lose));
      
    for (t in 1:trials){
      log_lik[t] = bernoulli_logit_lpmf(h[t] | winB * win +  loseB * lose);
    }
  
}


