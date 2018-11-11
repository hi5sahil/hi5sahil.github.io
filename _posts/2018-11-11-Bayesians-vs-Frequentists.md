---
title: Bayesians vs Frequentists!
date: "11 Nov 2018"
---

1. _**Frequentist View of Probability**_  
The probability of an event (ex - observing an odd number on rolling a six faced die) is its relative frequency among all posible outcomes (here - 6 possible outcomes) when performing large number of independent identical trials (rolling the die)

Note - The probability of an event for a single roll of die (just one trial) is either 0 (False) or 1 (True)

_Confidence Intervals_  
* A frequentists makes a H0 (null hypothesis) about the true value of parameter
* Collects random sample
* Calculates sample statistics & confidence intervals (ex - 95%)
* Makes a claim that the probability we have obtained a confidence interval which contains the true value of parameter is 95%
* All the while acknowleding that the parameter is fixed (but difficult to practically measure) and the randomness is coming from sampling (its the confidence interval that varies from sample to sample, not the population parameter)
* H0 is assumed to be true to begin with
* Then frequentists calculates the p-value assuming H0 is true
* If p-value (probability of getting a sample statistic as we did if H0 was true) is less than 0.05, we reject the H0 
* Final Conclusion is TRUE (fail to reject) or FALSE (reject)



2. _**Bayesian View of Probability**_  
In Bayesian world the probability of an event occuring is a subjective degree of belief 

Note - The probability of an event for a single roll of die (just one trial) can be 0 (False), 1 (True) or something in between (Plausible)

_Credible Intervals_  
* Unlike Frequentist Inference, Bayesian Inference doesn't necessarily needs many trials
* Bayesians start with a prior belief (from domain knowledge about the hypothesis space) and use evidence from observed data to update their prior belief to calculate posterior belief
* Bayesians then calculats the mean and sted dev of posterior distribution
* Makes a claim that one should reasonably believe (with 95% probability) that population parameter is within the credible interval
* Doesn't guarantees that 95% of the credible intervals (from 100 different samples) actually contain the true value of parameter
* Final Conclusion is TRUE, FALSE or Plausible

3. _**Bayesians' View is better when:**_  
* It is not possible to perform iid trials - 
What is the probabiliity France will win against Croatia in WC Final?
* Cold Start Problems - 
A clinical trial to find the right dose of a new medicine? (How to treat the first patient without any data? - use expert opinion / prior belief)
Recommendation for a new user on Spotify or Netflix?


