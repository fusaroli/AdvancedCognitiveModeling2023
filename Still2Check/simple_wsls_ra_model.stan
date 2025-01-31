

// Step 1 is to specify the input (i.e. the data for the model).
data {
 int<lower=1> n; // n is the number of trials in the data. It is specified as an integer with a lower boundary of 1 (as we cannot analyze a dataset with less than 1 trial.)
 array[n] int choice; // choice is a list containing the sequence of choices that the WSLS agent made (right hand is coded as 1, left hand as 0). The choice variable is specified as an array of integers that has the length of n.
 array[n] int F_c; // Frequency of choice. This is a feedback-like variable that tells us the frequency of the WSLS agent choosing the winning hand (1 for the winning hand, and -1 for the losing hand). It is specified as an array of integers that has the length of n.
}

// Step 2 is to specify the parameters that the model needs to estimate.
parameters {
  real bias; // Bias is the agent’s bias towards choosing the right hand. This parameter is a on a log-odds scale, so – 3 means always doing the opposite of your rule, and +3 means always following the rule.
  real beta; // Beta is the tendency of the WSLS agent to swith hand given that it loses a trial. Like bias, this  parameter is also on a log-odds scale.
}

// Step 3 is to specify the model to be estimated. In this case we are looking at a gaussian, with the parameters bias and beta, and priors on the bias and the beta.
model {
  // In the following two lines, we set the priors for our parameters. The priors for both of the parameters bias and beta are a gaussian distribution with a mean of 0 and an sd of 1. This means that the priors are relatively uninformed.
  target += normal_lpdf(bias | 0, 1);
  target += normal_lpdf(beta | 0, 1);
  
  // The model consists of a bernoulli distribution (binomial with just a single trial). The theta here is an expression of a linear model with bias as the intercept, beta as out slope, and the F_c variable as our x. This means that if beta is high then the model is deterministic, and if beta is close to 0.5 then the model is probablistic.
  target += bernoulli_logit_lpmf(choice | bias + beta * to_vector(F_c)); // We use the bernoulli_logit to reperameterize, i.e. we use math to change the geometry of the model and make the model move in the spaces we want. We use it because we have an outcome (bound between 0 and 1, since it is a probability) generated through a binomial. The bernoulli_logit is an inverse logit, meaning that it takes whatever we put into it and squeezes it into the 0-1 space. 
}

