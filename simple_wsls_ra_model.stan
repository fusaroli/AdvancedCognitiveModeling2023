
// Step 1 is to specify the input (i.e. the data for the model).
data {
 int<lower=1> n; // n is the number of trials in the data. It is specified as an integer with a lower boundary of 1 (as we cannot analyze a dataset with less than 1 trial.)
 array[n] int choice; // choice is a list containing the sequence of choices that the WSLS agent made (right hand is coded as 1, left hand as 0). The choice variable is specified as an array of integers that has the length of n.
 array[n] int F_c; // Frequency of choice. This is a feedback-like variable that tells us the frequency of the WSLS agent choosing the winning hand (1 for the winning hand, and -1 for the losing hand). It is specified as an array of integers that has the length of n. xxx
}
// Step 2 is to specify the parameters that the model needs to estimate.
parameters {
  real bias; // Bias is the rate at which the WSLS agent chooses the right (?xxx) hand. In our model this is determined by the otherBias (the bias) that we set for the random agent that the WSLS agent is playing against. This parameter is a probability and therefore bound between 0 and 1 (?xxx should we not specify lower and upper boundaries, then? <lower=0, upper=1>).
  real beta; // Beta is the tendency of the WSLS agent to swith hand given that it loses a trial. Like bias, this  parameter is also a probability bound between 0 and 1 (xxx)
}
// Step 3 is to specify the model to be estimated. In this case we are looking at a gaussian, with the parameters bias and beta, and priors on the bias and the beta.
model {
  // In the following two lines, we set the priors for our parameters. The priors for both of the parameters bias and beta are a gaussian distribution with a mean of 0 and an sd of 1. This means that the priors are relatively uninformed (xxx why do we want this?).
  target += normal_lpdf(bias | 0, 1);
  target += normal_lpdf(beta | 0, 1);
  // The model consists of a bernoulli distribution (binomial with just a single trial). The theta here is an expression of a linear model with bias as the intercept, beta as out slope, and the F_c variable as our x. This means that if beta is high then the model is deterministic, and if beta is close to 0.5 then the model is probablistic.
  target += bernoulli_logit_lpmf(choice | bias + beta * to_vector(F_c)); // We use the bernoulli_logit to reperameterize, i.e. we use math to change the geometry of the model and make the model move in the spaces we want. We use it because we have an outcome (bound between 0 and 1, since it is a probability) generated through a binomial. The bernoulli_logit is an inverse logit, meaning that it takes whatever we put into it and squeezes it into the 0-1 space. 
  // xxx Bernoulli_logit says: I assume that the rate is on log-odds scale, therefore I will (expand it?). The rate is on a log-odds scale. ...
}

