##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
### THIS CODE CREATES THE DIFFERENT FUNCTIONS TO PERFORM ESTIMATION OF THE LATENT CLASS   ####
### MODELS WITH COVARIATES UNDER THE ALGORITHMS DISCUSSED IN THE PAPER (INCLUDING THE NEW #### 
### NESTED EM AND THE HYBRID NESTED EM WE DEVELOP TO IMPROVE MAXIMIZATION PERFORMANCE).   ####
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################

#Load useful libraries
library(poLCA)
library(dummies)
library(nnet)

# GENERAL IMPORTANT NOTE: In the paper we fix the beta coefficients associated with the last 
# latent class R to 0, for identifiability. The functions below, fix instead the beta coefficients 
# associated with the first latent class r=1 to 0. We operate this choice here to be consistent 
# with the routines in the R library poLCA (for a potential future implementation of our algorithms 
# in this library). Since the classes are latent, the class labels are arbitrary, and are interpreted 
# only after estimation in the light of the class-specific probabilities of the categorical variables. 
# Therefore this operation is possible without loss of generality and interpretability.


##############################################################################################
##############################################################################################
#### EM ALGORITHM WITH NEWTON-RAPHSON STEPS ##################################################
##############################################################################################
##############################################################################################
# DESCRIPTION: This function performs one-step estimation using the EM algorithm with one 
# Newton-Raphson step as in the R library poLCA. The code also allows for a correction 
# (controlled by 0<alpha<1) to reduce the chance of decays in the log-likelihood sequence 
# (McLachlan and Krishnan 2007). When alpha=1, the function performs the routine with the classical 
# Newton-Raphson step without corrections as in Bandeen-Roche et al. (1997).

# DETAILS OF THE METHODOLOGY: This type of EM is carefully discussed in Section 1.1 of the paper.

# DETAILS OF THE INPUT QUANTITIES:
#- formula: defines the latent class model with covariates to be estimated (as in the poLCA library).
#- data: input data (including the multivariate categorical responses y and the covariates x in the formula).
#- nclass: number of latent classes to be considered in the model.
#- maxiter: maximum number of iterations in the EM to be considered.
#- tol: the EM algorithm stops when an additional iteration increases the log-likelihood by less than tol.
#- seed: seed to be considered when initializing the beta and pi parameters from random draws.
#- alpha: tuning parameter to rescale the updating of beta, and reduce concerns with drops in the log-likelihood sequence.

newton_em <- function(formula, data, nclass = 2, maxiter = 1000, tol = 1e-11, seed=1, alpha=1){
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# CREATE USEFUL QUANTITIES (following the notation in the paper)
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
mframe <- model.frame(formula, data)
y <- model.response(mframe)
x <- model.matrix(formula, mframe)
n <- nrow(y)
J <- ncol(y)
K.j <- t(matrix(apply(y, 2, max)))
R <- nclass
P <- ncol(x)

# dll denotes the variation in the log-likelihood sequence at iteration t, to decide when to stop the routine.
dll <- Inf

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# CREATE THE QUANTITIES TO BE SAVED
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# llik denotes the log-likelihood sequence
llik <- rep(NA,maxiter)
# llik_decrement is a vector monitoring drops in the log-likelihood sequence.
llik_decrement <- rep(NA,maxiter)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# INITIALIZE THE PARAMETERS
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
set.seed(seed)
#------------------------------------------------
# Conditional Probabilities (pi) ----------------
#------------------------------------------------
probs <- list()
for (j in 1:J) {
probs[[j]] <- matrix(runif(R * K.j[j]), nrow = R, ncol = K.j[j])
probs[[j]] <- probs[[j]]/rowSums(probs[[j]])}
probs.init <- probs
vp <- poLCA:::poLCA.vectorize(probs)

#------------------------------------------------
# Beta Coefficients (beta) ----------------------
#------------------------------------------------
b <- rnorm(P*(R - 1), 0, 0.1)	

#------------------------------------------------
# Log-likelihood at the initial values-----------
#------------------------------------------------
prior <- poLCA:::poLCA.updatePrior(b, x, R)

iter <- 1
llik[iter] <- sum(log(rowSums(prior * poLCA:::poLCA.ylik.C(vp,y))))
llik_decrement[iter] <- (dll < -1e-07)*1

#---------------------------------------------------------------------------------------------
##############################################################################################
# ALGORITHM
##############################################################################################
#---------------------------------------------------------------------------------------------
while ((iter < maxiter)  & (abs(dll) > tol)  & (!is.na(llik[iter]))){
#--------------------------------------------------------------------------------
iter <- iter + 1
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# E-STEP	
#--------------------------------------------------------------------------------
# Compute the conditional expectation of the latent class indicators for each unit
rgivy <- poLCA:::poLCA.postClass.C(prior, vp, y)

#--------------------------------------------------------------------------------
# M-STEP	
#--------------------------------------------------------------------------------
# M-STEP for pi:
# Maximize the probabilities of each categorical variable within every class
vp$vecprobs <- poLCA:::poLCA.probHat.C(rgivy, y, vp)

# M-STEP for beta:
# Maximize the beta coefficiens in the multinomial logit for the classes via Newton-Raphson
dd <- poLCA:::poLCA.dLL2dBeta.C(rgivy, prior, x)
b <- b + alpha*ginv(-dd$hess) %*% dd$grad
prior <- poLCA:::poLCA.updatePrior(b, x, R)

#--------------------------------------------------------------------------------
# UPDATE THE QUANTITIES TO BE MONITORED
#--------------------------------------------------------------------------------
#----------------------------------------------
# Log-Likelihood Sequence ---------------------
#----------------------------------------------
llik[iter] <- sum(log(rowSums(prior * poLCA:::poLCA.ylik.C(vp, y))))

#----------------------------------------------
# Decrement (0 if no decay, 1 if decay) -------
#----------------------------------------------
llik_decrement[iter] <- (dll < -1e-07)*1

#--------------------------------------------------------------------------------
# COMPUTE THE INCREMENT TO STUDY THE STATUS OF CONVERGENCE
#--------------------------------------------------------------------------------
dll <- llik[iter] - llik[iter - 1]}

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# OUTPUT OF THE ALGORITHM
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# NOTE: In the paper we focus on the computational properties of the algorithms and 
# therefore we output the number of iterations for convergence, the log-likelihood 
# sequence, and the decrements (if any) in the log-likelihood sequence. Clearly when 
# providing inference on the model, also the estimates of the parameters beta (b) and 
# the probabilities of the categorical variables y within each latent class (pi) should 
# be given as output.
output <- list()
output[[1]] <- iter
output[[2]] <- llik
output[[3]] <- llik_decrement

return(output)
}




 


##############################################################################################
##############################################################################################
### NESTED EM ALGORITHM ######################################################################
##############################################################################################
##############################################################################################
# DESCRIPTION: This function perform one-step estimation using our nested EM algorithm leveraging 
# the Pòlya-Gamma data augmentation described in the paper.

# DETAILS OF THE METHODOLOGY: This type of EM is carefully discussed in Section 2.2 of the paper.

# DETAILS OF THE INPUT QUANTITIES:
#- formula: defines the latent class model with covariates to be estimated (as in the poLCA library).
#- data: input data (including the multivariate categorical responses y and the covariates x in the formula).
#- nclass: number of latent classes to be considered in the model.
#- maxiter: maximum number of iterations in the EM to be considered.
#- tol: the EM algorithm stops when an additional iteration increases the log-likelihood by less than tol.
#- seed: seed to be considered when initializing the beta and pi parameters from random draws.

nested_em <- function(formula, data, nclass = 2, maxiter = 1000, tol = 1e-11, seed=1){
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# CREATE USEFUL QUANTITIES (following the notation in the paper)
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
mframe <- model.frame(formula, data)
y <- model.response(mframe)
x <- model.matrix(formula, mframe)
n <- nrow(y)
J <- ncol(y)
K.j <- t(matrix(apply(y, 2, max)))
R <- nclass
P <- ncol(x)

# dll denotes the variation in the log-likelihood sequence at iteration t, to decide when to stop the routine.
dll <- Inf

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# CREATE THE QUANTITIES TO BE SAVED
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# llik denotes the log-likelihood sequence
llik <- rep(NA,maxiter)
# llik_decrement is a vector monitoring drops in the log-likelihood sequence.
llik_decrement <- rep(NA,maxiter)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# INITIALIZE THE PARAMETERS
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
set.seed(seed)
#------------------------------------------------
# Conditional Probabilities (pi) ----------------
#------------------------------------------------
probs <- list()
for (j in 1:J) {
probs[[j]] <- matrix(runif(R * K.j[j]), nrow = R, ncol = K.j[j])
probs[[j]] <- probs[[j]]/rowSums(probs[[j]])}
probs.init <- probs
vp <- poLCA:::poLCA.vectorize(probs)

#------------------------------------------------
# Beta Coefficients (beta) ----------------------
#------------------------------------------------
b <- rnorm(P*(R - 1), 0, 0.1)	

#------------------------------------------------
# Log-likelihood at the initial values-----------
#------------------------------------------------
prior <- poLCA:::poLCA.updatePrior(b, x, R)

iter <- 1
llik[iter] <- sum(log(rowSums(prior * poLCA:::poLCA.ylik.C(vp,y))))
llik_decrement[iter] <- (dll < -1e-07)*1

#---------------------------------------------------------------------------------------------
##############################################################################################
# ALGORITHM
##############################################################################################
#---------------------------------------------------------------------------------------------
while ((iter < maxiter)  & (abs(dll) > tol)  & (!is.na(llik[iter]))){
#--------------------------------------------------------------------------------
iter <- iter + 1
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# E-STEP	
#--------------------------------------------------------------------------------
# Compute the conditional expectation of the latent class indicators for each unit
rgivy <- poLCA:::poLCA.postClass.C(prior, vp, y)

#--------------------------------------------------------------------------------
# M-STEP	
#--------------------------------------------------------------------------------
# M-STEP for pi:
# Maximize the probabilities of each categorical variable within every class
vp$vecprobs <- poLCA:::poLCA.probHat.C(rgivy, y, vp)

# M-STEP for beta:
# Maximize the beta coefficiens in the multinomial logit for the classes via Pòlya-Gamma
for (j in 2:R) {
               #----------------------------------------------------------------------------
               # Create the matrix of the current beta coefficients
               beta <- cbind(rep(0, P), matrix(b, P, R - 1))
               # Compute quantities a_i as defined in the paper
               a_i <- log(rowSums(exp(x %*% beta[, -j])))
               #----------------------------------------------------------------------------
               # NESTED E-STEP
               #----------------------------------------------------------------------------
               # Update the expectation of the latent class indicators for each unit
               E_s <- poLCA:::poLCA.postClass.C(poLCA:::poLCA.updatePrior(b, x, R),  vp, y)
               # Compute the expectation of the Pòlya-Gamma variables for each unit
               eta_j <- (x %*% beta[, j] - a_i)
               E_w <- 0.5 * as.double(tanh(0.5 * eta_j)/eta_j)
               #----------------------------------------------------------------------------
               # NESTED M-STEP
               #----------------------------------------------------------------------------
               # Update beta_j via Generalized Least Squares
               eta <- cbind((E_s[, j] - 0.5 + E_w * a_i)/E_w)
               beta[, j]  <-  solve(crossprod(x*sqrt(E_w)), crossprod(x,E_w*eta))
               b <- as.double(beta[, -1])
               }              
prior <- poLCA:::poLCA.updatePrior(b, x, R)

#--------------------------------------------------------------------------------
# UPDATE THE QUANTITIES TO BE MONITORED
#--------------------------------------------------------------------------------
#----------------------------------------------
# Log-Likelihood Sequence ---------------------
#----------------------------------------------
llik[iter] <- sum(log(rowSums(prior * poLCA:::poLCA.ylik.C(vp, y))))

#----------------------------------------------
# Decrement (0 if no decay, 1 if decay) -------
#----------------------------------------------
llik_decrement[iter] <- (dll < -1e-07)*1

#--------------------------------------------------------------------------------
# COMPUTE THE INCREMENT TO STUDY THE STATUS OF CONVERGENCE
#--------------------------------------------------------------------------------
dll <- llik[iter] - llik[iter - 1]}

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# OUTPUT OF THE ALGORITHM
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# NOTE: In the paper we focus on the computational properties of the algorithms and 
# therefore we output the number of iterations for convergence, the log-likelihood 
# sequence, and the decrements (if any) in the log-likelihood sequence. Clearly when 
# providing inference on the model, also the estimates of the parameters beta (b) 
# and the probabilities of the categorical variables y within each latent class (pi) 
# should be given as output.
output <- list()
output[[1]] <- iter
output[[2]] <- llik
output[[3]] <- llik_decrement

return(output)
}



 


##############################################################################################
##############################################################################################
### HYBRID NESTED EM ALGORITHM ###############################################################
##############################################################################################
##############################################################################################
# DESCRIPTION: This function performs one-step estimation using an hybrid version of the nested 
# EM proposed in Section 2.2. In particular the routine reaches a neighborhood of the maximum 
# using the more stable nested EM, and then switches to Newton-Raphson to speed convergence.

# DETAILS OF THE METHODOLOGY: This type of EM is carefully discussed in Section 3.3 of the paper.

# DETAILS OF THE INPUT QUANTITIES:
#- formula: defines the latent class model with covariates to be estimated (as in the poLCA library).
#- data: input data (including the multivariate categorical responses y and the covariates x in the formula).
#- nclass: number of latent classes to be considered in the model.
#- maxiter: maximum number of iterations in the EM to be considered.
#- tol: the EM algorithm stops when an additional iteration increases the log-likelihood by less than tol.
#- seed: seed to be considered when initializing the beta and pi parameters from random draws.
#- epsilon: the nested EM switches to Newton-Raphson when an iteration increases the log-likelihood by less than epsilon.

hybrid_em <- function(formula, data, nclass = 2, maxiter = 1000, tol = 1e-11, seed=1, epsilon=0.1){
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# CREATE USEFUL QUANTITIES (following the notation in the paper)
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
mframe <- model.frame(formula, data)
y <- model.response(mframe)
x <- model.matrix(formula, mframe)
n <- nrow(y)
J <- ncol(y)
K.j <- t(matrix(apply(y, 2, max)))
R <- nclass
P <- ncol(x)

# dll denotes the variation in the log-likelihood sequence at iteration t, to decide when to stop the routine.
dll <- Inf

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# CREATE THE QUANTITIES TO BE SAVED
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# llik denotes the log-likelihood sequence
llik <- rep(NA,maxiter)
# llik_decrement is a vector monitoring drops in the log-likelihood sequence.
llik_decrement <- rep(NA,maxiter)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# INITIALIZE THE PARAMETERS
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
set.seed(seed)
#------------------------------------------------
# Conditional Probabilities (pi) ----------------
#------------------------------------------------
probs <- list()
for (j in 1:J) {
probs[[j]] <- matrix(runif(R * K.j[j]), nrow = R, ncol = K.j[j])
probs[[j]] <- probs[[j]]/rowSums(probs[[j]])}
probs.init <- probs
vp <- poLCA:::poLCA.vectorize(probs)

#------------------------------------------------
# Beta Coefficients (beta) ----------------------
#------------------------------------------------
b <- rnorm(P*(R - 1), 0, 0.1)	

#------------------------------------------------
# Log-likelihood at the initial values-----------
#------------------------------------------------
prior <- poLCA:::poLCA.updatePrior(b, x, R)

iter <- 1
llik[iter] <- sum(log(rowSums(prior * poLCA:::poLCA.ylik.C(vp,y))))
llik_decrement[iter] <- (dll < -1e-07)*1

#---------------------------------------------------------------------------------------------
##############################################################################################
#ALGORITHM
##############################################################################################
#---------------------------------------------------------------------------------------------
while ((iter < maxiter)  & (abs(dll) > tol)  & (!is.na(llik[iter]))){
#--------------------------------------------------------------------------------
iter <- iter + 1
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# E-STEP	
#--------------------------------------------------------------------------------
# Compute the conditional expectation of the latent class indicators for each unit
rgivy <- poLCA:::poLCA.postClass.C(prior, vp, y)

#--------------------------------------------------------------------------------
# M-STEP	
#--------------------------------------------------------------------------------
# M-STEP for pi:
# Maximize the probabilities of each categorical variable within every class
vp$vecprobs <- poLCA:::poLCA.probHat.C(rgivy, y, vp)

# M-STEP for beta:
# Maximize the beta coefficiens in the multinomial logit for the classes via Pòlya-Gamma or Newton-Raphson
if (dll>epsilon) {
	       for (j in 2:R) {
               #----------------------------------------------------------------------------
               # Create the matrix of the current beta coefficients
               beta <- cbind(rep(0, P), matrix(b, P, R - 1))
               # Compute quantities a_i as defined in the paper
               a_i <- log(rowSums(exp(x %*% beta[, -j])))
               #----------------------------------------------------------------------------
               # NESTED E-STEP
               #----------------------------------------------------------------------------
               # Update the expectation of the latent class indicators for each unit
               E_s <- poLCA:::poLCA.postClass.C(poLCA:::poLCA.updatePrior(b, x, R),  vp, y)
               # Compute the expectation of the Pòlya-Gamma variables for each unit
               eta_j <- (x %*% beta[, j] - a_i)
               E_w <- 0.5 * as.double(tanh(0.5 * eta_j)/eta_j)
               #----------------------------------------------------------------------------
               # NESTED M-STEP
               #----------------------------------------------------------------------------
               # Update Beta_j via Generalized Least Squares
               eta <- cbind((E_s[, j] - 0.5 + E_w * a_i)/E_w)
               beta[, j]  <-  solve(crossprod(x*sqrt(E_w)), crossprod(x,E_w*eta))
               b <- as.double(beta[, -1])
               }} else {
dd <- poLCA:::poLCA.dLL2dBeta.C(rgivy, prior, x)
b <- b + ginv(-dd$hess) %*% dd$grad}      
prior <- poLCA:::poLCA.updatePrior(b, x, R)

#--------------------------------------------------------------------------------
# UPDATE THE QUANTITIES TO BE MONITORED
#--------------------------------------------------------------------------------
#----------------------------------------------
# Log-Likelihood Sequence ---------------------
#----------------------------------------------
llik[iter] <- sum(log(rowSums(prior * poLCA:::poLCA.ylik.C(vp, y))))

#----------------------------------------------
# Decrement (0 if no decay, 1 if decay) -------
#----------------------------------------------
llik_decrement[iter] <- (dll < -1e-07)*1

#--------------------------------------------------------------------------------
# COMPUTE THE INCREMENT TO STUDY THE STATUS OF CONVERGENCE
#--------------------------------------------------------------------------------
dll <- llik[iter] - llik[iter - 1]}

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# OUTPUT OF THE ALGORITHM
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# NOTE: In the paper we focus on the computational properties of the algorithms and 
# therefore we output the number of iterations for convergence, the log-likelihood 
# sequence, and the decrements (if any) in the log-likelihood sequence. Clearly when 
# providing inference on the model, also the estimates of the parameters beta (b) and 
# the probabilities of the categorical variables y within each latent class (pi) 
# should be given as output.
output <- list()
output[[1]] <- iter
output[[2]] <- llik
output[[3]] <- llik_decrement

return(output)
}




 


##############################################################################################
##############################################################################################
### EM FOR UNCONDITIONAL LATENT CLASS MODELS #################################################
##############################################################################################
##############################################################################################
# DESCRIPTION: EM algoritm for latent class analysis without covariates. As discussed in the paper, 
# this EM algorithm is useful for step 1 in the three--step estimation methods. Note that, to 
# exploit the optimized functions of the R library poLCA, we work with the log-odds reparameterization 
# b of the unconditional class probabilities nu.

# DETAILS OF THE METHODOLOGY: This type of EM is carefully discussed in step 1 of Section 1.2 in the paper.

# DETAILS OF THE INPUT QUANTITIES:
#- formula: defines the latent class model with covariates to be estimated (as in the poLCA library).
#- data: input data (including the multivariate categorical responses y and the covariates x in the formula).
#- nclass: number of latent classes to be considered in the model.
#- maxiter: maximum number of iterations in the EM to be considered.
#- tol: the EM algorithm stops when an additional iteration increases the log-likelihood by less than tol.
#- seed: seed to be considered when initializing the beta and pi parameters from random draws.

unconditional_em <- function(formula, data, nclass = 2, maxiter = 1000, tol = 1e-11, seed=1){
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# CREATE USEFUL QUANTITIES (following the notation in the paper)
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
mframe <- model.frame(formula, data)
y <- model.response(mframe)
x <- model.matrix(formula, mframe)
n <- nrow(y)
J <- ncol(y)
K.j <- t(matrix(apply(y, 2, max)))
R <- nclass
P <- ncol(x)

# dll denotes the variation in the log-likelihood sequence at iteration t, to decide when to stop the routine.
dll <- Inf

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# CREATE THE QUANTITIES TO BE SAVED
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# llik denotes the log-likelihood sequence
llik <- rep(NA,maxiter)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# INITIALIZE THE PARAMETERS
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
set.seed(seed)
#------------------------------------------------
# Conditional Probabilities (pi) ----------------
#------------------------------------------------
probs <- list()
for (j in 1:J) {
probs[[j]] <- matrix(runif(R * K.j[j]), nrow = R, ncol = K.j[j])
probs[[j]] <- probs[[j]]/rowSums(probs[[j]])}
probs.init <- probs
vp <- poLCA:::poLCA.vectorize(probs)

#------------------------------------------------
# Beta Coefficients (beta) ----------------------
#------------------------------------------------
b <- rep(0, P * (R - 1))	

#------------------------------------------------
# Log-likelihood at the initial values-----------
#------------------------------------------------
prior <- poLCA:::poLCA.updatePrior(b, x, R)

iter <- 1
llik[iter] <- sum(log(rowSums(prior * poLCA:::poLCA.ylik.C(vp,y))))

#---------------------------------------------------------------------------------------------
##############################################################################################
#ALGORITHM
##############################################################################################
#---------------------------------------------------------------------------------------------
while ((iter < maxiter)  & (abs(dll) > tol)  & (!is.na(llik[iter]))){
#--------------------------------------------------------------------------------
iter <- iter + 1
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# E-STEP	
#--------------------------------------------------------------------------------
# Compute the conditional expectation of the latent class indicators for each unit
rgivy <- poLCA:::poLCA.postClass.C(prior, vp, y)

#--------------------------------------------------------------------------------
# M-STEP	
#--------------------------------------------------------------------------------
# M-STEP for pi:
# Maximize analytically the probabilities of each categorical variable within every class
vp$vecprobs <- poLCA:::poLCA.probHat.C(rgivy, y, vp)

# M-STEP for b:
# Maximize analytically the log-odds of the unconditional latent class probabilities
b <- log((apply(rgivy,2,sum)/n)[2:R]/((apply(rgivy,2,sum)/n)[1]))

prior <- poLCA:::poLCA.updatePrior(b, x, R)

#--------------------------------------------------------------------------------
# UPDATE THE QUANTITIES TO BE MONITORED
#--------------------------------------------------------------------------------
#----------------------------------------------
# Log-Likelihood Sequence ---------------------
#----------------------------------------------
llik[iter] <- sum(log(rowSums(prior * poLCA:::poLCA.ylik.C(vp, y))))

#--------------------------------------------------------------------------------
# COMPUTE THE INCREMENT TO STUDY THE STATUS OF CONVERGENCE
#--------------------------------------------------------------------------------
dll <- llik[iter] - llik[iter - 1]}

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# OUTPUT OF THE ALGORITHM
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# NOTE: Here we output also the parameters estimates and the conditional probabilities 
# of the latent classes for each statistical unit at convergence since they are required 
# at steps 2 and 3 of the three--step methods. We do not monitor instead decays in the 
# log-likelihood sequence, since the EM for unconditional latent class models is a pure 
# EM algorithm.
output <- list()
output[[1]] <- iter
output[[2]] <- llik[iter]
output[[3]] <- vp
output[[4]] <- poLCA:::poLCA.postClass.C(prior, vp, y)
output[[5]] <- prior

return(output)
}




 


##############################################################################################
##############################################################################################
### EM FOR BIAS ADJUSTMENT IN THREE--STEP METHODS ############################################
##############################################################################################
##############################################################################################
# DESCRIPTION: Implement the EM algorithm for the bias correction proposed by Vermunt (2010) 
# in step 3 of the three-step estimating procedure for latent class models with covariates.

# DETAILS OF THE METHODOLOGY: This EM is discussed in Sections 1.2 and 4 in the paper, as well as in Vermunt (2010).

# DETAILS OF THE INPUT QUANTITIES:
#- formula: defines the latent class model with covariates to be estimated (as in the poLCA library).
#- data: input data (including the multivariate categorical responses y and the covariates x in the formula).
#- nclass: number of latent classes to be considered in the model.
#- maxiter: maximum number of iterations in the EM to be considered.
#- tol: the EM algorithm stops when an additional iteration increases the log-likelihood by less than tol.
#- seed: seed to be considered when initializing the beta and pi parameters from random draws.
#- classification_error: classification error table from step 2, required for bias correction.

correction_em <- function(formula, data, nclass = 2, maxiter = 1000, tol = 1e-11, seed=1, classification_error=class_err){
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# CREATE USEFUL QUANTITIES (following the notation in the paper)
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
mframe <- model.frame(formula, data)
y <- model.response(mframe)
x <- model.matrix(formula, mframe)
n <- nrow(x)
y <- matrix(y,n,1)
J <- ncol(y)
R <- nclass
P <- ncol(x)

# dll denotes the variation in the log-likelihood sequence at iteration t, to decide when to stop the routine.
dll <- Inf

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# CREATE THE QUANTITIES TO BE SAVED
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# llik denotes the log-likelihood sequence
llik <- rep(NA,maxiter)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# INITIALIZE THE PARAMETERS
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
set.seed(seed)
#------------------------------------------------
# Conditional Probabilities (pi) ----------------
#------------------------------------------------
# NOTE: These quantities are fixed during all the routine, since the EM required in step 3 
# of the correction procedure proposed by Vermunt (2010) has a single categorical response 
# whose probabilities within each class coincides with the classification error probabilities 
# computed from step 2.
probs <- list()
probs[[1]] <- classification_error
probs.init <- probs
vp <- poLCA:::poLCA.vectorize(probs)

#------------------------------------------------
# Beta Coefficients (beta) ----------------------
#------------------------------------------------
b <- rnorm(P*(R - 1), 0, 0.1)	

#------------------------------------------------
# Log-likelihood at the initial values-----------
#------------------------------------------------
prior <- poLCA:::poLCA.updatePrior(b, x, R)

iter <- 1
llik[iter] <- sum(log(rowSums(prior * poLCA:::poLCA.ylik.C(vp,y))))

#---------------------------------------------------------------------------------------------
##############################################################################################
#ALGORITHM
##############################################################################################
#---------------------------------------------------------------------------------------------
while ((iter < maxiter)  & (abs(dll) > tol)  & (!is.na(llik[iter]))){
#--------------------------------------------------------------------------------
iter <- iter + 1
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# E-STEP	
#--------------------------------------------------------------------------------
# Compute the conditional expectation of the latent class indicators for each unit
rgivy <- poLCA:::poLCA.postClass.C(prior, vp, y)

#--------------------------------------------------------------------------------
# M-STEP	
#--------------------------------------------------------------------------------
# Only the coefficient beta require maximization here. We use Newton-Raphson as in LATENT GOLD.
# Maximize the beta coefficiens in the multinomial logit for the classes via Newton-Raphson
dd <- poLCA:::poLCA.dLL2dBeta.C(rgivy, prior, x)
b <- b + ginv(-dd$hess) %*% dd$grad
prior <- poLCA:::poLCA.updatePrior(b, x, R)

#--------------------------------------------------------------------------------
# UPDATE THE QUANTITIES TO BE MONITORED
#--------------------------------------------------------------------------------
#----------------------------------------------
# Log-Likelihood Sequence ---------------------
#----------------------------------------------
llik[iter] <- sum(log(rowSums(prior * poLCA:::poLCA.ylik.C(vp, y))))

#--------------------------------------------------------------------------------
# COMPUTE THE INCREMENT TO STUDY THE STATUS OF CONVERGENCE
#--------------------------------------------------------------------------------
dll <- llik[iter] - llik[iter - 1]}

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# OUTPUT OF THE ALGORITHM
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# NOTE: Here we output also the parameters estimates for b to then evaluate the 
# full--model log-likelihood for comparison with one-step methods.
output <- list()
output[[1]] <- iter
output[[2]] <- llik
output[[3]] <- b

return(output)
}
