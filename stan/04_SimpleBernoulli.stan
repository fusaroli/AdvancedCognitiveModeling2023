
// Stan Model: Simple Bernoulli for Bias Estimation
// Goal: Estimate the underlying probability (theta) of choosing 1 ('right')
//       given a sequence of binary choices.

// 1. Data Block: Declares the data Stan expects from R
data {
  int<lower=1> n;        // Number of trials (must be at least 1)
  array[n] int<lower=0, upper=1> h; // Array 'h' of length 'n' containing choices (0 or 1)
}

// 2. Parameters Block: Declares the parameters the model will estimate
parameters {
  real<lower=0, upper=1> theta; // The bias parameter (probability), constrained between 0 and 1
}

// 3. Model Block: Defines the priors and the likelihood
model {
  // Prior: Our belief about theta *before* seeing the data.
  // We use a Beta(1, 1) prior, which is equivalent to a Uniform(0, 1) distribution.
  // This represents maximal prior ignorance about the bias.
  // 'target +=' adds the log-probability density to the overall model log-probability.
  target += beta_lpdf(theta | 1, 1); // lpdf = log probability density function

  // Likelihood: How the data 'h' depend on the parameter 'theta'.
  // We model each choice 'h' as a Bernoulli trial with success probability 'theta'.
  // The model assesses how likely the observed sequence 'h' is given a value of 'theta'.
  target += bernoulli_lpmf(h | theta); // lpmf = log probability mass function (for discrete data)
}

// 4. Generated Quantities Block (Optional but useful)
// Code here is executed *after* sampling, using the estimated parameter values.
// Useful for calculating derived quantities or predictions.
generated quantities {
  // Example: Simulate a new dataset based on the estimated theta
  array[n] int h_pred = bernoulli_rng(rep_vector(theta, n)); // _rng = random number generation
}

