
// Mixture model for binary choice data
// Combines a biased choice process with a random choice process

data {
  int<lower=1> n;  // Number of trials
  array[n] int h;  // Choice data (0/1)
}

parameters {
  real bias;        // Bias parameter for biased process (logit scale)
  real noise_logit; // Mixing weight for random process (logit scale)
}

model {
  // Priors
  target += normal_lpdf(bias | 0, 1);          // Prior for bias parameter (centered at 0.5 in prob scale)
  target += normal_lpdf(noise_logit | -1, 1);  // Prior for noise proportion (favors lower noise)
  
  // Mixture likelihood using log_sum_exp for numerical stability
  target += log_sum_exp(
    log(inv_logit(noise_logit)) +            // Log probability of random process
    bernoulli_logit_lpmf(h | 0),             // Log likelihood under random process (p=0.5)
    
    log1m(inv_logit(noise_logit)) +          // Log probability of biased process
    bernoulli_logit_lpmf(h | bias)           // Log likelihood under biased process
  );
}

generated quantities {
  // Transform parameters to probability scale for easier interpretation
  real<lower=0, upper=1> noise_p = inv_logit(noise_logit);  // Proportion of random choices
  real<lower=0, upper=1> bias_p = inv_logit(bias);          // Bias toward right in biased choices
  
  // Predicted distributions
  vector[n] log_lik;
  array[n] int pred_component;  // Which component generated each prediction (1=random, 0=biased)
  array[n] int pred_choice;     // Predicted choices
  
  // Calculate log likelihood for each observation (for model comparison)
  for (i in 1:n) {
    log_lik[i] = log_sum_exp(
      log(noise_p) + bernoulli_logit_lpmf(h[i] | 0),
      log1m(noise_p) + bernoulli_logit_lpmf(h[i] | bias)
    );
  }
  
  // Generate posterior predictions
  for (i in 1:n) {
    // First determine which component to use
    pred_component[i] = bernoulli_rng(noise_p);
    
    // Then generate prediction from appropriate component
    if (pred_component[i] == 1) {
      // Random component
      pred_choice[i] = bernoulli_rng(0.5);
    } else {
      // Biased component
      pred_choice[i] = bernoulli_rng(bias_p);
    }
  }
}

