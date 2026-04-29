
// Model B: Marginalised Mixture over Fixed Candidate Rules
// This is NOT the cognitive theory (Model A, the particle filter).
// It is a tractable proxy that admits HMC validation.
// Validation of Model B tests Model B, not Model A.
//
// Architecture (identical in principle to Ch. 8 mixture models):
//   - Discrete latent variable = which rule the agent follows
//   - Marginalized analytically with log_sum_exp over a fixed rule set
//   - Only continuous parameter: logit_error_prob (unconstrained)
//   - error_prob = inv_logit(logit_error_prob)
//
// No sequential learning in this model: it treats all trials as conditionally
// exchangeable given error_prob and the rule mixture.
//
// The 7 rules below cover the main decision structures visible in the
// Kruschke stimulus layout, including two nested rules of the form
// A op (B inner C) that cannot be expressed as flat 2D conjunctions/disjunctions.
// See §'Discharging the Post-Hoc Rule Selection' for the sensitivity analysis.

data {
  int<lower=1> ntrials;
  int<lower=1> nfeatures;                       // must be 2 for this rule set
  array[ntrials] int<lower=0, upper=1> y;       // observed choices
  array[ntrials, nfeatures] real obs;           // obs[,1]=Height, obs[,2]=Position

  // Prior hyperparameters (passed as data for transparency)
  real prior_logit_error_mean;
  real<lower=0> prior_logit_error_sd;
}

parameters {
  real logit_error_prob;   // unconstrained; error_prob = inv_logit(logit_error_prob)
}

transformed parameters {
  real<lower=0, upper=1> error_prob = inv_logit(logit_error_prob);

  // Per-trial choice probability under each rule, computed once per leapfrog
  // step so log_lik in generated quantities is a one-line lookup.
  // (Same architectural discipline as Ch 11 GCM and Ch 12 prototype.)
  array[ntrials] real lp_marginal;

  for (i in 1:ntrials) {
    real h = obs[i, 1];
    real p = obs[i, 2];

    // -- Flat 1D and 2D rules (same as original 5) --------------------------
    int pred1 = h > 2.5 ? 1 : 0;
    int pred2 = p < 2.5 ? 1 : 0;
    int pred3 = (h > 2.0 && p > 3.0) ? 1 : 0;
    int pred4 = (h < 2.0 || p < 2.0) ? 0 : 1;
    int pred5 = (h > 2.5 && p < 3.5) ? 1 : 0;
    // -- Nested rules: A op (B inner C) -------------------------------------
    // pred6: (h > 3.5) OR ((h > 1.5) AND (p < 1.5))
    // Cognitive reading: very tall OR some height AND very low position
    // Perfectly classifies the 8 Kruschke stimuli.
    int pred6 = (h > 3.5 || (h > 1.5 && p < 1.5)) ? 1 : 0;
    // pred7: (h > 1.5) AND ((p < 2.5) OR (h > 3.5))
    // Cognitive reading: some height AND low position OR very tall
    // Also perfectly classifies the 8 Kruschke stimuli.
    int pred7 = (h > 1.5 && (p < 2.5 || h > 3.5)) ? 1 : 0;

    int R = 7;

    vector[R] lp;
    lp[1] = bernoulli_lpmf(y[i] | pred1 == 1 ? 1 - error_prob : error_prob);
    lp[2] = bernoulli_lpmf(y[i] | pred2 == 1 ? 1 - error_prob : error_prob);
    lp[3] = bernoulli_lpmf(y[i] | pred3 == 1 ? 1 - error_prob : error_prob);
    lp[4] = bernoulli_lpmf(y[i] | pred4 == 1 ? 1 - error_prob : error_prob);
    lp[5] = bernoulli_lpmf(y[i] | pred5 == 1 ? 1 - error_prob : error_prob);
    lp[6] = bernoulli_lpmf(y[i] | pred6 == 1 ? 1 - error_prob : error_prob);
    lp[7] = bernoulli_lpmf(y[i] | pred7 == 1 ? 1 - error_prob : error_prob);

    lp_marginal[i] = log_sum_exp(lp) - log(R);
  }
}

model {
  target += normal_lpdf(logit_error_prob | prior_logit_error_mean, prior_logit_error_sd);
  for (i in 1:ntrials) target += lp_marginal[i];
}

generated quantities {
  // log_lik[i] is just a lookup of lp_marginal[i] -- no recomputation.
  array[ntrials] real log_lik;
  for (i in 1:ntrials) log_lik[i] = lp_marginal[i];

  real lprior = normal_lpdf(logit_error_prob | prior_logit_error_mean, prior_logit_error_sd);
}

