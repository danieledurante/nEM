Performance assessment for the ELECTION data
================
Daniele Durante

Description
================

This tutorial implementation focuses on assessing the maximization performance and the computational efficiency of the different algorithms for the estimation of latent class models with covariates. In particular, this assessment considers the dataset `election` from the `R` library [`poLCA`](https://www.jstatsoft.org/article/view/v042i10).

The analyses reproduce those discussed in Section 3.1 of the paper: [Durante, D., Canale, A. and Rigon, T. (2017). *A nested expectation-maximization algorithm for latent class models with covariates* \[arXiv:1705.03864\]](https://arxiv.org/abs/1705.03864), where we propose a novel **nested EM** algorithm for improved maximum likelihood estimation of latent class models with covariates.

Load the source functions and the data
================

The functions for the implementation of the different algorithms (including the **nested EM**, and the popular competitors currently considered in the literature) are available in the source file [`LCA-Covariates-Algorithms.R`](https://github.com/danieledurante/nEM/blob/master/LCA-Covariates-Algorithms.R). More comments on the different maximization routines can be found in the file `LCA-Covariates-Algorithms.R`.

Let us load this source file, along with the `election` dataset.

``` r
rm(list = ls())
source("LCA-Covariates-Algorithms.R")
data(election)
str(election[, c(1:12,17)])
```

    ## 'data.frame':    1785 obs. of  13 variables:
    ##  $ MORALG : Factor w/ 4 levels "1 Extremely well",..: 3 4 1 2 2 2 2 2 2 1 ...
    ##  $ CARESG : Factor w/ 4 levels "1 Extremely well",..: 1 3 2 2 4 3 NA 2 2 1 ...
    ##  $ KNOWG  : Factor w/ 4 levels "1 Extremely well",..: 2 4 2 2 2 3 2 2 2 1 ...
    ##  $ LEADG  : Factor w/ 4 levels "1 Extremely well",..: 2 3 1 2 3 2 2 2 3 1 ...
    ##  $ DISHONG: Factor w/ 4 levels "1 Extremely well",..: 3 2 3 2 2 2 4 3 4 4 ...
    ##  $ INTELG : Factor w/ 4 levels "1 Extremely well",..: 2 2 2 2 2 2 2 2 2 1 ...
    ##  $ MORALB : Factor w/ 4 levels "1 Extremely well",..: 1 NA 2 2 1 2 NA 3 2 2 ...
    ##  $ CARESB : Factor w/ 4 levels "1 Extremely well",..: 1 NA 2 3 1 NA 3 4 3 4 ...
    ##  $ KNOWB  : Factor w/ 4 levels "1 Extremely well",..: 2 2 2 2 2 3 2 4 2 2 ...
    ##  $ LEADB  : Factor w/ 4 levels "1 Extremely well",..: 2 3 3 2 2 2 2 4 2 3 ...
    ##  $ DISHONB: Factor w/ 4 levels "1 Extremely well",..: 4 NA 3 3 3 2 4 3 3 3 ...
    ##  $ INTELB : Factor w/ 4 levels "1 Extremely well",..: 2 1 2 1 2 2 NA 4 2 2 ...
    ##  $ PARTY  : num  5 3 1 3 7 1 6 1 1 1 ...

Since the main focus is on comparing the computational performance of the different algorithms, let us for simplicity remove all the statistical units having missing values.

``` r
election <- na.omit(election)
```

Consistent with the analyses in Section 3.1 of the paper: [Durante, D., Canale, A. and Rigon, T. (2017). *A nested expectation-maximization algorithm for latent class models with covariates* \[arXiv:1705.03864\]](https://arxiv.org/abs/1705.03864), we focus on the latent class model for the `J = 12` categorical evaluations of the candidates Al Gore and George Bush, with the variable `PARTY` (affiliation party) as a covariate in the multinomial logistic regression for the latent classes. Using the syntax of the library `poLCA`, this model can be defined as follows:

``` r
f_election <- cbind(MORALG, CARESG, KNOWG, LEADG, DISHONG, INTELG, 
                    MORALB, CARESB, KNOWB, LEADB, DISHONB, INTELB) ~ PARTY
```

Estimation with R = 2 latent classes
================

We perform estimation of the parameters in the above latent class model with covariates under different computational routines (including our novel **nested EM** algorithm), and compare maximization performance along with computational efficiency.

Consistent with the tutorial analyses in [Linzer and Lewis (2011)](https://www.jstatsoft.org/article/view/v042i10), we first focus on the model with `R = 2` classes.

``` r
n_c <- 2
```

To provide a detailed computational assessment, we perform estimation under the different algorithms for `Rep_Tot = 100` runs at varying initializations. For each run, the algorithms are all initialized at the same starting values, which are controlled by a `seed` (changing across the different runs). Let us, therefore, define this 100 seeds and the variance `sigma` of the zero mean Gaussian variables from which the initial values for the *β* parameters are generated.

``` r
Rep_Tot <- 100
seed_rep <- c(1:100)
sigma <- 0.5
```

---
#### 1. EM algorithm with Newton-Raphson as in Bandeen-Roche et al. (1997) (one-step)

Here we consider the one-step EM algorithm with Newton-Raphson methods proposed by [Bandeen-Roche et al. (1997)](https://www.jstor.org/stable/2965407), and discussed in Section 1.1 of our paper. This requires the function `newton_em()` in the source file `LCA-Covariates-Algorithms.R` we uploaded before.

Let us first create the quantities to be monitored for each run. These include the number of iterations to reach convergence, the log-likelihood sequence, and a vector monitoring presence (1) or absence (0) of drops in the log-likelihood sequence at each iteration.

``` r
# Create the allocation matrices for the quantities to be monitored.
iter_NR_EM <- rep(0, Rep_Tot)
llik_NR_EM <- matrix(0, Rep_Tot, 1000)
llik_decrement_NR_EM <- matrix(0, Rep_Tot, 1000)

# 1000 means that the maximum number of iterations in the EM algorithm is fixed to 1000. 
```

Finally, let us perform the `Rep_Tot = 100` runs of the one-step EM algorithm with Newton-Raphson methods. We also monitor the computational time via the function `system.time()`.

``` r
# Perform the algorithm.
time_NR_EM <- system.time(  
for (rep in 1:Rep_Tot){
fit_NR_EM <- newton_em(f_election, election, nclass = n_c, seed = seed_rep[rep],sigma_init=sqrt(sigma))
iter_NR_EM[rep] <- fit_NR_EM[[1]]	
llik_NR_EM[rep,] <- fit_NR_EM[[2]]
llik_decrement_NR_EM[rep,] <- fit_NR_EM[[3]]}
)[3]
```

---
#### 2. EM algorithm with Newton-Raphson as in Formann (1992) and Van der Heijden et al. (1996) (one-step)

As discussed in the paper, the algorithm proposed by [Bandeen-Roche et al. (1997)](https://www.jstor.org/stable/2965407) leverages a Newton-Raphson step which is based on a plug-in estimate for the Hessian of the full-model log-likelihood, instead of maximizing the expectation of the complete log-likelihood. A more formal approach implemented in [Formann (1992)](http://www.jstor.org/stable/2290280?seq=1#page_scan_tab_contents) and [Van der Heijden et al. (1996)](https://www.jstor.org/stable/1165269?seq=1#page_scan_tab_contents), considers instead a Newton-Raphson step aimed at maximizing a quadratic approximation of the expectation for the complete log-likelihood.  Implementation of this routine requires requires the function `newton_em_Q1()` in the source file `LCA-Covariates-Algorithms.R` we uploaded before.

``` r
# Create the allocation matrices for the quantities to be monitored.
iter_NR_EM_Q1 <- rep(0, Rep_Tot)
llik_NR_EM_Q1 <- matrix(0, Rep_Tot, 1000)
llik_decrement_NR_EM_Q1 <- matrix(0, Rep_Tot, 1000)

# Perform the algorithm.
time_NR_EM_Q1 <- system.time(	
for (rep in 1:Rep_Tot){
fit_NR_EM_Q1 <- newton_em_Q1(f_election, election, nclass = n_c, seed = seed_rep[rep],sigma_init=sqrt(sigma))
iter_NR_EM_Q1[rep] <- fit_NR_EM_Q1[[1]]	
llik_NR_EM_Q1[rep,] <- fit_NR_EM_Q1[[2]]
llik_decrement_NR_EM_Q1[rep,] <- fit_NR_EM_Q1[[3]]}
)[3]

```

---
#### 3. Re-scaled EM algorithm with Newton-Raphson (one-step)

Here we consider the re-scaled version of the above one-step EM algorithm with Newton-Raphson methods. This modification is discussed in Section 1.1 of our paper, and its general version can be found in Chapter 1.5.6 of [McLachlan and Krishnan (2007)](http://onlinelibrary.wiley.com/book/10.1002/9780470191613). Also this algorithm requires the function `newton_em_Q1()` in the source file `LCA-Covariates-Algorithms.R` we uploaded before. However, now the parameter 0 &lt; *α* &lt; 1 should be modified to reduce concerns about drops in the log-likelihood sequence. Here we consider, the case with *α* = 0.5.

``` r
# Create the allocation matrices for the quantities to be monitored.
iter_NR_EM_Q1_0.5 <- rep(0, Rep_Tot)
llik_NR_EM_Q1_0.5 <- matrix(0, Rep_Tot, 1000)
llik_decrement_NR_EM_Q1_0.5 <- matrix(0, Rep_Tot, 1000)

# Perform the algorithm.
time_NR_EM_Q1_0.5 <- system.time(	
for (rep in 1:Rep_Tot){
fit_NR_EM_Q1_0.5 <- newton_em_Q1(f_election, election, alpha=0.5,nclass = n_c, seed = seed_rep[rep],sigma_init=sqrt(sigma))
iter_NR_EM_Q1_0.5[rep] <- fit_NR_EM_Q1_0.5[[1]]	
llik_NR_EM_Q1_0.5[rep,] <- fit_NR_EM_Q1_0.5[[2]]
llik_decrement_NR_EM_Q1_0.5[rep,] <- fit_NR_EM_Q1_0.5[[3]]}
)[3]
```

---
#### 4. MM-EM algorithm based on the lower-bound routine in Bohning (1992) (one-step)

All the previous algorithms are not guaranteed to provide monotone log-likelihood sequences. A more robust routine proposed by [Bohning (1992)](https://link.springer.com/article/10.1007/BF00048682) replaces the Hessian in the above Newton-Raphson updates, with a matrix that dominates it globally, thus guaranteeing monotone and more stable convergence. Although  [Bohning (1992)](https://link.springer.com/article/10.1007/BF00048682) focuses on a multinomial logistic regression with observed responses, their procedure can be adapted also to our latent class model with covariates as seen in the function `MM_em()` in the source file `LCA-Covariates-Algorithms.R`.

``` r
# Create the allocation matrices for the quantities to be monitored.
iter_MM <- rep(0, Rep_Tot)
llik_MM <- matrix(0, Rep_Tot, 1000)
llik_decrement_MM <- matrix(0, Rep_Tot, 1000)

# Perform the algorithm.
time_MM <- system.time(	 
for (rep in 1:Rep_Tot){
fit_MM <- MM_em(f_election, election, nclass = n_c, seed = seed_rep[rep],sigma_init=sqrt(sigma))
iter_MM[rep] <- fit_MM[[1]]	
llik_MM[rep,] <- fit_MM[[2]]
llik_decrement_MM[rep,] <- fit_MM[[3]]}
)[3]
```

---
#### 5. Classical 3-step algorithm (three-step)

Here we consider the classical three-step strategy to estimate latent class models with covariates (e.g. [Clogg 1995](https://www.iser.essex.ac.uk/research/publications/494549)). As discussed in Section 1.2 of our paper, this algorithm consists of three steps.

1.  Estimate a latent class model without covariates `f_cheating_unconditional <- cbind(MORALG, CARESG, KNOWG, LEADG, DISHONG, INTELG, MORALB, CARESB, KNOWB, LEADB, DISHONB, INTELB) ~ 1`. This requires the function `unconditional_em()` (in `LCA-Covariates-Algorithms.R`).
2.  Using the estimates in 1, predict the latent classes *s*<sub>*i*</sub>, *i* = 1, ..., *n*, by assigning each unit *i* to the class *r* with the highest predicted probability.
3.  Using the `R` function `multinom` in the library `nnet`, estimate the coefficients *β*<sub>1</sub>, ..., *β*<sub>*R*</sub> from a multinomial logistic regression with the predicted *s*<sub>1</sub>, ..., *s*<sub>*n*</sub> as responses.

The code to implement this routine and save the relevant quantities is:

``` r
# Create the allocation matrix for the full--model log-likelihood sequence.
llik_3_step_classical <- rep(0, Rep_Tot)

# Define the useful quantities to compute the full--model log-likelihood sequence.
f_election_3_step <- cbind(MORALG, CARESG, KNOWG, LEADG, DISHONG, INTELG, 
                           MORALB, CARESB, KNOWB, LEADB, DISHONB, INTELB) ~ PARTY
nclass = n_c
mframe_election_3_step <- model.frame(f_election_3_step, election)
y_election_3_step <- model.response(mframe_election_3_step)
x_election_3_step <- model.matrix(f_election_3_step, mframe_election_3_step)
R_election_3_step <- nclass

# Perform the three step algorithm.
time_3_step_classical <- system.time( 
for (rep in 1:Rep_Tot){
#---------------------------------------------------------------------------------------------------
# 1] Estimate a latent class model without covariates
#---------------------------------------------------------------------------------------------------
f_election_unconditional <- cbind(MORALG, CARESG, KNOWG, LEADG, DISHONG, INTELG, 
                                  MORALB, CARESB, KNOWB, LEADB, DISHONB, INTELB) ~ 1
fit_unconditional <- unconditional_em(f_election_unconditional, election, nclass = n_c, seed = seed_rep[rep])

#---------------------------------------------------------------------------------------------------
# 2] Predict the latent class of each unit via modal assignment
#---------------------------------------------------------------------------------------------------
pred_class <- apply(fit_unconditional[[4]],1, which.max)

#---------------------------------------------------------------------------------------------------
# 3] Estimate the beta coefficients from a multinomial logit with the predicted classes as responses
#---------------------------------------------------------------------------------------------------
b <- c(t(summary(multinom(pred_class ~ election$PARTY, trace = FALSE))$coefficients))

# Compute the log-likelihood of the full model
prior <- poLCA:::poLCA.updatePrior(b, x_election_3_step, R_election_3_step)
llik_3_step_classical[rep] <- sum(log(rowSums(prior * poLCA:::poLCA.ylik.C(fit_unconditional[[3]], 
                                  y_election_3_step))))}
)[3]
```

---
#### 6. Bias-corrected 3-step algorithm (three-step)

Here we implement the modification proposed by [Vermunt (2010)](https://academic.oup.com/pan/article-abstract/18/4/450/1518615/Latent-Class-Modeling-with-Covariates-Two-Improved) of the classical three-step methods, in order to reduce the bias of the estimators. This strategy is discussed in Sections 1.2 of our paper, and proceed as follows:

1.  Estimate a latent class model without covariates `f_cheating_unconditional <- cbind(MORALG, CARESG, KNOWG, LEADG, DISHONG, INTELG, MORALB, CARESB, KNOWB, LEADB, DISHONB, INTELB) ~ 1`. This requires the function `unconditional_em()` (in `LCA-Covariates-Algorithms.R)`.
2.  Using the estimates in 1, predict the latent classes *s*<sub>*i*</sub>, *i* = 1, ..., *n*, by assigning each unit *i* to the class *r* with the highest predicted probability. Compute also the classification error by applying equation (6) in [Vermunt (2010)](https://academic.oup.com/pan/article-abstract/18/4/450/1518615/Latent-Class-Modeling-with-Covariates-Two-Improved).
3.  Following equation (19) in [Vermunt (2010)](https://academic.oup.com/pan/article-abstract/18/4/450/1518615/Latent-Class-Modeling-with-Covariates-Two-Improved) estimate the coefficients *β*<sub>1</sub>, ..., *β*<sub>*R*</sub> from a latent class model with covariates, where the predicted latent classes from 2 act as the only categorical variable available, and its probability mass function within each class is fixed and equal to the classification error. This implementation requires the function `correction_em()` in `LCA-Covariates-Algorithms.R`.

The code to implement this routine and save the relevant quantities is:

``` r
# Create the allocation matrix for the full--model log-likelihood sequence.
llik_3_step_corrected <- rep(0, Rep_Tot)

# Define the useful quantities to compute the full--model log-likelihood sequence.
f_election_3_step <- cbind(MORALG, CARESG, KNOWG, LEADG, DISHONG, INTELG, 
                           MORALB, CARESB, KNOWB, LEADB, DISHONB, INTELB) ~ PARTY
nclass = n_c
mframe_cheating_3_step <- model.frame(f_election_3_step, election)
y_election_3_step <- model.response(mframe_election_3_step)
x_election_3_step <- model.matrix(f_election_3_step, mframe_election_3_step)
R_election_3_step <- nclass

# Perform the three step algorithm.
time_3_step_corrected <- system.time(  
for (rep in 1:Rep_Tot){
#---------------------------------------------------------------------------------------------------
# 1] Estimate a latent class model without covariates
#---------------------------------------------------------------------------------------------------
f_election_unconditional <- cbind(MORALG, CARESG, KNOWG, LEADG, DISHONG, INTELG, 
                                  MORALB, CARESB, KNOWB, LEADB, DISHONB, INTELB) ~ 1
fit_unconditional <- unconditional_em(f_election_unconditional, election, nclass = n_c, seed = seed_rep[rep])

#---------------------------------------------------------------------------------------------------
# 2] Predict the latent class of each unit via modal assignment and compute the classification error
#---------------------------------------------------------------------------------------------------
pred_class <- apply(fit_unconditional[[4]], 1, which.max)
class_err <- matrix(0, nclass, nclass)
rownames(class_err) <- paste("W", c(1:nclass), sep="")
colnames(class_err) <- paste("X", c(1:nclass), sep="")
for (r in 1:nclass){
class_err[,r] <- (t(dummy(pred_class))%*%as.matrix(fit_unconditional[[4]][,r],
                  dim(election)[1],1)/dim(election)[1])/fit_unconditional[[5]][1,r]}
class_err <- t(class_err)

#---------------------------------------------------------------------------------------------------
# 3] Estimate the beta coefficients from the correction procedure in Vermunt (2010)
#---------------------------------------------------------------------------------------------------
f_election_3_step_correct <- cbind(pred_class) ~ PARTY
fit_correct <- correction_em(f_election_3_step_correct, election, nclass = n_c, seed = seed_rep[rep], 
                             classification_error = class_err,sigma_init=0.1)

# Compute the log-likelihood of the full model
prior <- poLCA:::poLCA.updatePrior(fit_correct[[3]], x_election_3_step, R_election_3_step)
llik_3_step_corrected[rep] <- sum(log(rowSums(prior * poLCA:::poLCA.ylik.C(fit_unconditional[[3]], 
                                  y_election_3_step))))}
)[3]
```
---
#### 7. Nested EM algorithm (one-step)

We now implement our **nested EM** algorithm for improved one-step estimation of latent class models with covariates. This routine is carefully described in Section 2.2 of our paper, and leverages the recently developed Pòlya-Gamma data augmentation ([Polson et al. 2013](http://www.tandfonline.com/doi/abs/10.1080/01621459.2013.829001)). The implementation requires the function `nested_em()` in the source file `LCA-Covariates-Algorithms.R`.

``` r
# Create the allocation matrices for the quantities to be monitored.
iter_NEM <- rep(0, Rep_Tot)
llik_NEM <- matrix(0, Rep_Tot, 1000)
llik_decrement_NEM <- matrix(0, Rep_Tot, 1000)

# Perform the algorithm.
time_NEM <- system.time(	
for (rep in 1:Rep_Tot){
fit_NEM <- nested_em(f_election, election, nclass = n_c, seed = seed_rep[rep],sigma_init=sqrt(sigma))
iter_NEM[rep] <- fit_NEM[[1]]	
llik_NEM[rep,] <- fit_NEM[[2]]
llik_decrement_NEM[rep,] <- fit_NEM[[3]]}
)[3]
```
---
#### 8. Hybrid nested EM algorithm (one-step)

Here we consider a more efficient hybrid version of the **nested EM** algorithm which reaches a neighborhood of the maximum using the more stable **nested EM**, and then switches to Newton-Raphson methods to speed convergence. This routine is described in Section 2.3 of our paper, and requires the function `hybrid_em()` in the source file `LCA-Covariates-Algorithms.R`.

``` r
# Create the allocation matrices for the quantities to be monitored.
iter_HYB <- rep(0, Rep_Tot)
llik_HYB <- matrix(0, Rep_Tot, 1000)
llik_decrement_HYB <- matrix(0, Rep_Tot, 1000)

# Perform the algorithm.
time_HYB <- system.time( 
for (rep in 1:Rep_Tot){
fit_HYB <- hybrid_em(f_election, election, nclass = n_c, seed = seed_rep[rep], epsilon = 0.01,sigma_init=sqrt(sigma))
iter_HYB[rep] <- fit_HYB[[1]]	
llik_HYB[rep,] <- fit_HYB[[2]]
llik_decrement_HYB[rep,] <- fit_HYB[[3]]}
)[3]
```

Performance comparison
----------------------

Once the parameters have been estimated under the computational routines implemented above, we compare the maximization performance and the computational efficiency of the different algorithms, in order to reproduce the results in Table 1 of our paper: [Durante, D., Canale, A. and Rigon, T. (2017). *A nested expectation-maximization algorithm for latent class models with covariates* \[arXiv:1705.03864\]](https://arxiv.org/abs/1705.03864). In particular, we consider the following quantities, computed for each run of every routine:

**Maximization Performance**

-   Number of runs with a drop in the log-likelihood sequence.
-   Number of runs converging to values which are not the maximum log-likelihood.
-   For the runs reaching a local mode, we also compute the quartiles of the difference between the log-likelihoods in the local modes and the maximum one.

**Computational Efficiency**

-   Quartiles of the number of iterations for convergence, computed only for the runs reaching the maximum.
-   Averaged computational time for each run.

Consistent with the above goal, let us create a function which computes the measures of performance from the output of the different algorithms.

``` r
performance_algo <- function(max_loglik, n_rep, loglik_seq, loglik_decay, n_iter, time, delta){

#----------------------------------------------------------------------------------------------
# Number of runs with a drop in the log-likelihood sequence.
#----------------------------------------------------------------------------------------------  
n_drops <- 0
for (rep in 1:n_rep){
n_drops <- n_drops + (sum(loglik_decay[rep,1:n_iter[rep]]) > 0)*1}

#----------------------------------------------------------------------------------------------
# Number of runs converging to values which are not the maximum log-likelihood.
#---------------------------------------------------------------------------------------------- 
n_l_modes <- sum(abs(max_loglik - loglik_seq[cbind(1:n_rep,n_iter)]) > delta)

#----------------------------------------------------------------------------------------------
# Quartiles of the difference between log-likelihoods in local modes and the maximum one.
#----------------------------------------------------------------------------------------------
sel_modes <- which(abs(max_loglik - loglik_seq[cbind(1:n_rep,n_iter)]) > delta)
diff_llik <- quantile(abs(max_loglik - loglik_seq[cbind(1:n_rep,n_iter)])[sel_modes])[2:4]

#----------------------------------------------------------------------------------------------
# Quartiles of the number of iterations to reach convergence to maximum log-likelihood.
#----------------------------------------------------------------------------------------------
sel_convergence <- which(abs(max_loglik - loglik_seq[cbind(1:n_rep,n_iter)]) <= delta)
iter_convergence <- quantile(n_iter[sel_convergence])[2:4]

#----------------------------------------------------------------------------------------------
# Averaged computational time for each iteration.
#----------------------------------------------------------------------------------------------
averaged_time <- time/n_rep

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#Create the vector with the measures to be saved
output<-c(n_drops, n_l_modes, diff_llik, iter_convergence, averaged_time)
return(output)
}
```

In reproducing the results in Table 1, let us first define the correct maximum log-likelihood `max_llik`, and a control quantity `delta` defining the minimum deviation from `max_llik` which is indicative of a local mode.

``` r
max_llik <- c(-11102.72)
delta <- 0.01
```

Finally, let us create a matrix `Table_Performance` which contains the performance measures for the different algorithms.

``` r
Table_Performance <- matrix(NA,9,8)

rownames(Table_Performance) <- c("N. Decays",
                                 "N. Local Modes",
                                 "Q1 Log-L in Local Modes",
                                 "Q2 Log-L in Local Modes",
                                 "Q3 Log-L in Local Modes",
                                 "Q1 N. Iterat. Converge max(Log-L)",
                                 "Q2 N. Iterat. Converge max(Log-L)",
                                 "Q3 N. Iterat. Converge max(Log-L)",
                                 "Averaged Time")

colnames(Table_Performance) <- c("NR EM","NR EM Q1","NR EM Q1 0.5","MM EM",
                                 "CLASSIC. 3-STEP","CORREC. 3-STEP","NESTED EM","HYBRID EM")
```

We can now compute the different performance measures for our algorithms.

**1. Performance EM algorithm with Newton-Raphson methods *α* = 1 (one-step maximization)**

``` r
Table_Performance[,1] <- performance_algo(max_llik, Rep_Tot, llik_NR_EM_alpha_1, 
                                          llik_decrement_NR_EM_alpha_1, iter_NR_EM_alpha_1, 
                                          time_NR_EM_alpha_1, delta)
```

**2.1 Performance EM algorithm with Newton-Raphson methods *α* = 0.75 (one-step maximization)**

``` r
Table_Performance[,2] <- performance_algo(max_llik, Rep_Tot, llik_NR_EM_alpha_0.75, 
                                          llik_decrement_NR_EM_alpha_0.75, iter_NR_EM_alpha_0.75, 
                                          time_NR_EM_alpha_0.75, delta)
```

**2.2 Performance EM algorithm with Newton-Raphson methods *α* = 0.5 (one-step maximization)**

``` r
Table_Performance[,3] <- performance_algo(max_llik, Rep_Tot, llik_NR_EM_alpha_0.5, 
                                          llik_decrement_NR_EM_alpha_0.5, iter_NR_EM_alpha_0.5, 
                                          time_NR_EM_alpha_0.5, delta)
```

**2.3 Performance EM algorithm with Newton-Raphson methods *α* = 0.25 (one-step maximization)**

``` r
Table_Performance[,4] <- performance_algo(max_llik, Rep_Tot, llik_NR_EM_alpha_0.25, 
                                          llik_decrement_NR_EM_alpha_0.25, iter_NR_EM_alpha_0.25, 
                                          time_NR_EM_alpha_0.25, delta)
```

**3 Performance Classical 3-step algorithm (three-step maximization)** 
> As discussed in the paper, since all the three-step runs converge systematically to local modes, we do not study the number of iterations to reach convergence. In fact, these routines never converge to the maximum log-likelihood. Also the number of drops in the log-likelihood sequence is somewhat irrelevant to evaluate the three-step methods, since the estimation routines are based on two separate maximizations in steps 1 and 3, not directly related to the full-model log-likelihood.

``` r
Table_Performance[,5] <- performance_algo(max_llik, Rep_Tot, llik_3_step_classical, 
                                          matrix(0,Rep_Tot,1000), rep(0,Rep_Tot), 
                                          time_3_step_classical, delta)
Table_Performance[1,5] <- NA
```

**4 Performance Bias-corrected 3-step algorithm (three-step maximization)** 
> Even in this case all the runs converge systematically to local modes. Therefore we do not study the number of iterations to reach convergence. Also the number of drops in the log-likelihood sequence is somewhat irrelevant to evaluate the three-step methods, since the estimation routines are based on two separate maximizations in steps 1 and 3, not directly related to the full-model log-likelihood.

``` r
Table_Performance[,6] <- performance_algo(max_llik, Rep_Tot, llik_3_step_corrected, 
                                          matrix(0,Rep_Tot,1000), rep(0,Rep_Tot), 
                                          time_3_step_corrected, delta)
Table_Performance[1,6] <- NA
```

**5 Performance nested EM algorithm (one-step maximization)**

``` r
Table_Performance[,7] <- performance_algo(max_llik, Rep_Tot, llik_NEM, llik_decrement_NEM, 
                                          iter_NEM, time_NEM, delta)
```

**6 Performance hybrid nested EM algorithm (one-step maximization)**

``` r
Table_Performance[,8] <- performance_algo(max_llik, Rep_Tot, llik_HYB, llik_decrement_HYB, 
                                          iter_HYB, time_HYB, delta)
```

Analysis of the output from the table
-------------------------------------

Let us finally visualize the performance table, which reproduces Table 2 in the paper: [Durante, D., Canale, A. and Rigon, T. (2017). *A nested expectation-maximization algorithm for latent class models with covariates* \[arXiv:1705.03864\]](https://arxiv.org/abs/1705.03864).

In particular, the maximization performance and the computational efficiency of the EM algorithm with one Newton-Raphson step, along with those of the re-scaled modifications, are:

``` r
library(knitr)
kable(Table_Performance[,1:4])
```

|                                   |     NR EM 1|  NR EM 0.75|   NR EM 0.5|   NR EM 0.25|
|-----------------------------------|-----------:|-----------:|-----------:|------------:|
| N. Decays                         |    50.00000|    27.00000|   14.000000|    5.0000000|
| N. Local Modes                    |    60.00000|    44.00000|   26.000000|   12.0000000|
| Q1 Log-L in Local Modes           |   463.01375|   164.12244|    9.144528|    0.6445302|
| Q2 Log-L in Local Modes           |   770.20659|   640.60664|  330.426575|    9.1445285|
| Q3 Log-L in Local Modes           |  1142.95098|   968.03192|  561.004121|  300.2153311|
| Q1 N. Iterat. Converge max(Log-L) |   133.75000|   131.00000|  136.750000|  151.7500000|
| Q2 N. Iterat. Converge max(Log-L) |   169.50000|   174.50000|  183.500000|  209.0000000|
| Q3 N. Iterat. Converge max(Log-L) |   240.25000|   245.50000|  266.000000|  320.2500000|
| Averaged Time                     |     0.12613|     0.17079|    0.195310|    0.2215400|

The maximization performance and the computational efficiency of the three-step estimation algorithms, along with those of our **nested EM** and its hybrid modification, are instead:

``` r
library(knitr)
kable(Table_Performance[,5:8])
```

|                                   |  CLASSIC. 3-STEP|  CORREC. 3-STEP|    NESTED EM|    HYBRID EM|
|-----------------------------------|----------------:|---------------:|------------:|------------:|
| N. Decays                         |               NA|              NA|    0.0000000|    0.0000000|
| N. Local Modes                    |        100.00000|       100.00000|    9.0000000|    9.0000000|
| Q1 Log-L in Local Modes           |         42.22420|        39.10380|    0.1070165|    0.1070165|
| Q2 Log-L in Local Modes           |         42.22421|        39.10381|    0.6445302|    0.6445302|
| Q3 Log-L in Local Modes           |         42.23407|        39.10381|    6.0102387|    8.2221241|
| Q1 N. Iterat. Converge max(Log-L) |               NA|              NA|  146.0000000|  122.0000000|
| Q2 N. Iterat. Converge max(Log-L) |               NA|              NA|  182.0000000|  166.0000000|
| Q3 N. Iterat. Converge max(Log-L) |               NA|              NA|  257.5000000|  249.0000000|
| Averaged Time                     |          0.20476|         0.17436|    0.3427100|    0.2166100|

Reproduce the right plot in Figure 2 of the paper
-------------------------------------------------

We conclude the analysis by providing the code to obtain the right plot in Figure 2 of the paper: [Durante, D., Canale, A. and Rigon, T. (2017). *A nested expectation-maximization algorithm for latent class models with covariates* \[arXiv:1705.03864\]](https://arxiv.org/abs/1705.03864). This plot compares, for a selected run `sel <- 22`, the log-likelihood sequence obtained under our **nested EM** with the one provided by the standard EM algorithm with Newton-Raphson methods proposed by [Bandeen-Roche et al. (1997)](https://www.jstor.org/stable/2965407).

Let us first load some useful libraries and choose the run on which to focus.

``` r
library(ggplot2)
library(reshape)

sel <- 22
```

We then create a dataset `data_plot` with two columns. The first contains the log-likelihood sequence of the **nested EM**, whereas the second comprises the one from the standard EM algorithm with Newton-Raphson methods.

``` r
data_plot <- cbind(llik_NEM[sel,],llik_NR_EM_alpha_1[sel,])
data_plot <- data_plot[c(1:max(iter_NEM[sel],iter_NR_EM_alpha_1[sel])),]
data_plot[c((iter_NR_EM_alpha_1[sel] + 1):
          max(iter_NEM[sel],iter_NR_EM_alpha_1[sel])),2] <- data_plot[iter_NR_EM_alpha_1[sel],2]
```

Finally we create the Figure.

``` r
data_ggplot <- melt(data_plot)
data_ggplot <- as.data.frame(data_ggplot)
data_ggplot$app <- "ELECTION DATA"

plot <- ggplot(data = data_ggplot, aes(x = X1, y = value, group = X2)) +
               geom_line(aes(linetype = as.factor(X2))) +
               coord_cartesian(ylim = c(-15000, -10600), xlim = c(0, 50)) + theme_bw() +
               labs(x = "iteration", y = expression(paste(l,"(", theta^{(t)}, ";",x,",",y,")"))) +
               theme(legend.position = "none", strip.text.x = element_text(size = 11, face = "bold")) +
               facet_wrap(~ app)

plot
```

![](https://github.com/danieledurante/nEM/blob/master/Application%20to%20Election%20Data/right-plot-Figure-2.png)
