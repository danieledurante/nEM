##############################################################################
## LOAD DATA AND SOURCE FUNCTIONS ############################################
##############################################################################

# Load source file, along with the cheating dataset.
rm(list = ls())
source("LCA-Covariates-Algorithms.R")
data(cheating)
str(cheating)

# For simplicity remove all the statistical units having missing values.
cheating <- na.omit(cheating)

# Latent class model with covariates to be estimated.
f_cheating <- cbind(LIEEXAM, LIEPAPER, FRAUD, COPYEXAM) ~ GPA

# Define seeds of the 100 runs for each algorithm
Rep_Tot <- 100
seed_rep <- c(101:200)
seed_rep[35] <- 1
seed_rep[89] <- 2

##############################################################################
## RUN ALGORITHMS  ###########################################################
##############################################################################
#----------------------------------------------------------------------------------------
# EM + Newton-Raphson alpha=1
iter_NR_EM_alpha_1 <- rep(0, Rep_Tot)
llik_NR_EM_alpha_1 <- matrix(0, Rep_Tot, 1000)
llik_decrement_NR_EM_alpha_1 <- matrix(0, Rep_Tot, 1000)

time_NR_EM_alpha_1 <- system.time(  
for (rep in 1:Rep_Tot){
fit_NR_EM <- newton_em(f_cheating, cheating, nclass = 2, seed = seed_rep[rep])
iter_NR_EM_alpha_1[rep] <- fit_NR_EM[[1]]	
llik_NR_EM_alpha_1[rep,] <- fit_NR_EM[[2]]
llik_decrement_NR_EM_alpha_1[rep,] <- fit_NR_EM[[3]]}
)[3]

#----------------------------------------------------------------------------------------
# EM + Newton-Raphson alpha=0.75
iter_NR_EM_alpha_0.75 <- rep(0, Rep_Tot)
llik_NR_EM_alpha_0.75 <- matrix(0, Rep_Tot, 1000)
llik_decrement_NR_EM_alpha_0.75 <- matrix(0, Rep_Tot, 1000)

time_NR_EM_alpha_0.75 <- system.time(	  
for (rep in 1:Rep_Tot){
fit_NR_EM <- newton_em(f_cheating, cheating, nclass = 2, seed = seed_rep[rep], alpha = 0.75)
iter_NR_EM_alpha_0.75[rep] <- fit_NR_EM[[1]]	
llik_NR_EM_alpha_0.75[rep,] <- fit_NR_EM[[2]]
llik_decrement_NR_EM_alpha_0.75[rep,] <- fit_NR_EM[[3]]}
)[3]

#----------------------------------------------------------------------------------------
# EM + Newton-Raphson alpha=0.5
iter_NR_EM_alpha_0.5 <- rep(0, Rep_Tot)
llik_NR_EM_alpha_0.5 <- matrix(0, Rep_Tot, 1000)
llik_decrement_NR_EM_alpha_0.5 <- matrix(0, Rep_Tot, 1000)

time_NR_EM_alpha_0.5 <- system.time(	  
for (rep in 1:Rep_Tot){
fit_NR_EM <- newton_em(f_cheating, cheating, nclass = 2, seed = seed_rep[rep], alpha = 0.5)
iter_NR_EM_alpha_0.5[rep] <- fit_NR_EM[[1]]	
llik_NR_EM_alpha_0.5[rep,] <- fit_NR_EM[[2]]
llik_decrement_NR_EM_alpha_0.5[rep,] <- fit_NR_EM[[3]]}
)[3]

#----------------------------------------------------------------------------------------
# EM + Newton-Raphson alpha=0.25
iter_NR_EM_alpha_0.25 <- rep(0, Rep_Tot)
llik_NR_EM_alpha_0.25 <- matrix(0, Rep_Tot, 1000)
llik_decrement_NR_EM_alpha_0.25 <- matrix(0, Rep_Tot, 1000)

time_NR_EM_alpha_0.25 <- system.time(	  
for (rep in 1:Rep_Tot){
fit_NR_EM <- newton_em(f_cheating, cheating, nclass = 2, seed = seed_rep[rep], alpha = 0.25)
iter_NR_EM_alpha_0.25[rep] <- fit_NR_EM[[1]]	
llik_NR_EM_alpha_0.25[rep,] <- fit_NR_EM[[2]]
llik_decrement_NR_EM_alpha_0.25[rep,] <- fit_NR_EM[[3]]}
)[3]

#----------------------------------------------------------------------------------------
# 3-step Classical
llik_3_step_classical <- rep(0, Rep_Tot)
f_cheating_3_step <- cbind(LIEEXAM, LIEPAPER, FRAUD, COPYEXAM) ~ GPA
nclass = 2
mframe_cheating_3_step <- model.frame(f_cheating_3_step, cheating)
y_cheating_3_step <- model.response(mframe_cheating_3_step)
x_cheating_3_step <- model.matrix(f_cheating_3_step, mframe_cheating_3_step)
R_cheating_3_step <- nclass

time_3_step_classical <- system.time(
  
for (rep in 1:Rep_Tot){
#---------------------------------------------------------------------------------------------------
# 1] Estimate a latent class model without covariates
#---------------------------------------------------------------------------------------------------
f_cheating_unconditional <- cbind(LIEEXAM, LIEPAPER, FRAUD, COPYEXAM) ~ 1
fit_unconditional <- unconditional_em(f_cheating_unconditional, 
                                      cheating, nclass = 2, seed = seed_rep[rep])
#---------------------------------------------------------------------------------------------------
# 2] Predict the latent class of each unit via modal assignment
#---------------------------------------------------------------------------------------------------
pred_class <- apply(fit_unconditional[[4]],1, which.max)

#---------------------------------------------------------------------------------------------------
# 3] Estimate the beta coefficients from a multinomial logit with the predicted classes as responses
#---------------------------------------------------------------------------------------------------
b <- c(t(summary(multinom(pred_class ~ cheating$GPA, trace = FALSE))$coefficients))

# Compute the log-likelihood of the full model
prior <- poLCA:::poLCA.updatePrior(b, x_cheating_3_step, R_cheating_3_step)
llik_3_step_classical[rep] <- sum(log(rowSums(prior * poLCA:::poLCA.ylik.C(fit_unconditional[[3]],
                                  y_cheating_3_step))))}

)[3]

#----------------------------------------------------------------------------------------
# 3-step Correct
llik_3_step_corrected <- rep(0, Rep_Tot)
f_cheating_3_step <- cbind(LIEEXAM, LIEPAPER, FRAUD, COPYEXAM) ~ GPA
nclass = 2
mframe_cheating_3_step <- model.frame(f_cheating_3_step, cheating)
y_cheating_3_step <- model.response(mframe_cheating_3_step)
x_cheating_3_step <- model.matrix(f_cheating_3_step, mframe_cheating_3_step)
R_cheating_3_step <- nclass

time_3_step_corrected <- system.time(
  
for (rep in 1:Rep_Tot){
#---------------------------------------------------------------------------------------------------
# 1] Estimate a latent class model without covariates
#---------------------------------------------------------------------------------------------------
f_cheating_unconditional <- cbind(LIEEXAM, LIEPAPER, FRAUD, COPYEXAM) ~ 1
fit_unconditional <- unconditional_em(f_cheating_unconditional, 
                                      cheating, nclass = 2, seed = seed_rep[rep])

#---------------------------------------------------------------------------------------------------
# 2] Predict the latent class of each unit via modal assignment and compute the classification error
#---------------------------------------------------------------------------------------------------
pred_class <- apply(fit_unconditional[[4]], 1, which.max)
class_err <- matrix(0, nclass, nclass)
rownames(class_err) <- paste("W", c(1:nclass), sep="")
colnames(class_err) <- paste("X", c(1:nclass), sep="")
for (r in 1:nclass){
class_err[,r] <- (t(dummy(pred_class))%*%as.matrix(fit_unconditional[[4]][,r],
                  dim(cheating)[1],1)/dim(cheating)[1])/fit_unconditional[[5]][1,r]}
class_err <- t(class_err)

#---------------------------------------------------------------------------------------------------
# 3] Estimate the beta coefficients from the correction procedure in Vermunt (2010)
#---------------------------------------------------------------------------------------------------
f_cheating_3_step_correct <- cbind(pred_class) ~ GPA
fit_correct <- correction_em(f_cheating_3_step_correct, cheating, seed = seed_rep[rep],
                             classification_error = class_err)

# Compute the log-likelihood of the full model
prior <- poLCA:::poLCA.updatePrior(fit_correct[[3]], x_cheating_3_step, R_cheating_3_step)
llik_3_step_corrected[rep] <- sum(log(rowSums(prior * poLCA:::poLCA.ylik.C(fit_unconditional[[3]],
                                  y_cheating_3_step))))}

)[3]


#----------------------------------------------------------------------------------------
# Nested EM
iter_NEM <- rep(0, Rep_Tot)
llik_NEM <- matrix(0, Rep_Tot, 1000)
llik_decrement_NEM <- matrix(0, Rep_Tot, 1000)

time_NEM <- system.time(	  
for (rep in 1:Rep_Tot){
fit_NEM <- nested_em(f_cheating, cheating, nclass = 2, seed = seed_rep[rep])
iter_NEM[rep] <- fit_NEM[[1]]	
llik_NEM[rep,] <- fit_NEM[[2]]
llik_decrement_NEM[rep,] <- fit_NEM[[3]]}
)[3]

#----------------------------------------------------------------------------------------
# Hybrid Nested EM
iter_HYB <- rep(0, Rep_Tot)
llik_HYB <- matrix(0, Rep_Tot, 1000)
llik_decrement_HYB <- matrix(0, Rep_Tot, 1000)

time_HYB <- system.time(  
for (rep in 1:Rep_Tot){
fit_HYB <- hybrid_em(f_cheating, cheating, nclass = 2, seed = seed_rep[rep], epsilon = 0.1)
iter_HYB[rep] <- fit_HYB[[1]]	
llik_HYB[rep,] <- fit_HYB[[2]]
llik_decrement_HYB[rep,] <- fit_HYB[[3]]}
)[3]

##############################################################################
## PERFORMANCE ASSESSMENTS  (TABLE 1) ########################################
##############################################################################

#Create function for performance assessments
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


#Quantities for finding modes
max_llik <- c(-429.6384)
delta <- 0.01

#Create performance table
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

colnames(Table_Performance) <- c("NR EM 1","NR EM 0.75","NR EM 0.5","NR EM 0.25","CLASSIC. 3-STEP",
                                 "CORREC. 3-STEP","NESTED EM","HYBRID EM")

#----------------------------------------------------------------------------------------
# Compute the performance measures for the different algorithms
Table_Performance[,1] <- performance_algo(max_llik, Rep_Tot, llik_NR_EM_alpha_1, 
                                          llik_decrement_NR_EM_alpha_1, iter_NR_EM_alpha_1, 
                                          time_NR_EM_alpha_1, delta)

Table_Performance[,2] <- performance_algo(max_llik, Rep_Tot, llik_NR_EM_alpha_0.75, 
                                          llik_decrement_NR_EM_alpha_0.75, iter_NR_EM_alpha_0.75, 
                                          time_NR_EM_alpha_0.75, delta)

Table_Performance[,3] <- performance_algo(max_llik, Rep_Tot, llik_NR_EM_alpha_0.5, 
                                          llik_decrement_NR_EM_alpha_0.5, iter_NR_EM_alpha_0.5, 
                                          time_NR_EM_alpha_0.5, delta)

Table_Performance[,4] <- performance_algo(max_llik, Rep_Tot, llik_NR_EM_alpha_0.25, 
                                          llik_decrement_NR_EM_alpha_0.25, iter_NR_EM_alpha_0.25, 
                                          time_NR_EM_alpha_0.25, delta)

Table_Performance[,5] <- performance_algo(max_llik, Rep_Tot, llik_3_step_classical, 
                                          matrix(0,Rep_Tot,1000), rep(0,Rep_Tot), 
                                          time_3_step_classical, delta)
Table_Performance[1,5] <- NA


Table_Performance[,6] <- performance_algo(max_llik, Rep_Tot, llik_3_step_corrected, 
                                          matrix(0,Rep_Tot,1000), rep(0,Rep_Tot), 
                                          time_3_step_corrected, delta)
Table_Performance[1,6] <- NA

Table_Performance[,7] <- performance_algo(max_llik, Rep_Tot, llik_NEM, llik_decrement_NEM, 
                                          iter_NEM, time_NEM, delta)

Table_Performance[,8] <- performance_algo(max_llik, Rep_Tot, llik_HYB, llik_decrement_HYB, 
                                          iter_HYB, time_HYB, delta)
                                          
#----------------------------------------------------------------------------------------
# Visualize Table 1
Table_Performance[,1:4]
Table_Performance[,5:8]


##############################################################################
## REPRODUCE LEFT PLOT FIGURE 2 ##############################################
##############################################################################
library(ggplot2)
library(reshape)

sel <- 24

data_plot <- cbind(llik_NEM[sel,],llik_NR_EM_alpha_1[sel,])
data_plot <- data_plot[c(1:max(iter_NEM[sel],iter_NR_EM_alpha_1[sel])),]
data_plot[c((iter_NR_EM_alpha_1[sel] + 1):
          max(iter_NEM[sel],iter_NR_EM_alpha_1[sel])),2] <- data_plot[iter_NR_EM_alpha_1[sel],2]

data_ggplot <- melt(data_plot)
data_ggplot <- as.data.frame(data_ggplot)
data_ggplot$app <- "CHEATING DATA"

plot <- ggplot(data = data_ggplot, aes(x = X1, y = value, group = X2)) +
               geom_line(aes(linetype = as.factor(X2))) +
               coord_cartesian(ylim = c(-800, -420), xlim = c(0, 50)) + theme_bw() +
               labs(x = "iteration", y = expression(paste(l,"(", theta^{(t)}, ";",x,",",y,")"))) +
               theme(legend.position = "none", strip.text.x = element_text(size = 11, face = "bold")) +
               facet_wrap(~ app)

plot