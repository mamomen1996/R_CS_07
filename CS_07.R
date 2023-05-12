# Case-Study Title: Marketing Campaign Analysis via Logistic Regression (Data Driven approaches in Business and Marketing)
# Data Analysis methodology: CRISP-DM
# Dataset: Bundling strategy A/B testing data of an Iranian Telecom Company
# Case Goal: Evaluation of a pricing strategy (bundling) in business (Deduction from data)


### Required Libraries ----
#No library required


### Read Data from File ----
data <- read.csv('CS_07.csv', header = T)
dim(data)  # 3156 records, 4 variables


### Step 1: Business Understanding ----
 # know business process and issues
 # know the context of the problem
 # know the order of numbers in the business


### Step 2: Data Understanding ----
### Step 2.1: Data Inspection (Data Understanding from Free Perspective) ----
## Dataset variables definition
colnames(data)

#IFBundle:	does the coupon have bundle (do we offer bundle to customer)?		-> predictor variable
#IFUseCoupon:	does customer use discount coupon or not?				-> outcome variable
#Channel:	sending channel of coupon that sent to customer				-> controller variable
#no:		coupon number that sent to customer


### Step 2.2: Data Exploring (Data Understanding from Statistical Perspective) ----
## Overview of Dataframe
head(data)
tail(data)
str(data)  # all of variables in this case are Categorical
summary(data)

## Categorical variables should be stored as factor
data$Channel <- factor(data$Channel, levels = c('Mail', 'InPerson', 'Email'))
data$IFBundle <- factor(data$IFBundle, levels = c('NoBundle', 'Bundle'))
data$IFUseCoupon <- factor(data$IFUseCoupon, levels = c('No', 'Yes'))

summary(data)
#we have good distribution in categorical variables

## Univariate Profiling (check each variable individually)
#check to sure that have good distribution in each category
table(data$Channel)
table(data$IFBundle)
table(data$IFUseCoupon)

## Bivariate Profiling (measure 2-2 relationships between variables)
# Checking data balancing (having enough and balanced data in each category) -> must be > 30
table(data$IFUseCoupon, data$Channel)
table(data$IFUseCoupon, data$IFBundle)
table(data$Channel, data$IFBundle)
table(data$Channel, data$IFBundle, data$IFUseCoupon)

# Cross Tabulation Analysis
cross_tab <- table(data$IFBundle, data$IFUseCoupon)  # our hypotheses is that `IFBundle` has impact on `IFUseCoupon` rate
cross_tab

prop.table(cross_tab)  # total proportion
prop.table(cross_tab, 1)  # proportion over rows
prop.table(cross_tab, 2)  # proportion over columns

#Chi-Square Test for Cross Tabulation
chisq.test(cross_tab)
#H0: IFUseCoupon is independent of IFBundle
#H1: IFUseCoupon is related to IFBundle
#If p-value < 0.05 reject H0

#Descriptive Analysis result: offering bundle has positive impact on incitement customers to use their coupon and buy -> bundling is a successful strategy for this company
#but in Descriptive Analysis we can not quantify how much has impact


### Step 4: Modeling ----
# Predictive Analysis: use a Predictive model for deduction from data.
#we don't care to prediction here (predict y base on some x), we want to know impact of offering bundle in coupon on increasing customer purchase probability.

# Model 1:
m1 <- glm(IFUseCoupon ~ IFBundle, data = data, family = 'binomial')  # Logistic Regression

summary(m1)
#our created model should decrease Deviance if it was good model -> should be 'Residual Deviance' < 'Null Deviance'

#Deviance chi-squared test
#H0: the model is not better than chance at predicting the outcome
#p-value < 0.05 reject H0
modelChi1 <- m1$null.deviance - m1$deviance  # test statistic
Chidf1 <- m1$df.null - m1$df.residual  # chi-squared degree of freedom
Chisq_prob1 <- 1 - pchisq(modelChi1, Chidf1)  # p-value
Chisq_prob1

#result: deviance decreased, because our model was better than base model

#Assessing the contribution of predictors
#Wald test results: consider coefficient's p-values -> Significant Ceofficient -> yes, the variable is effective on explanation of Y

#Interpre Coefficients of model
exp(0.38879)  # Odds -> our point-estimate about IFBundle impact on increasing the purchase likelihood is 47.5% by average
#so, if be IFBundle == 'Yes' -> the bundle increases the purchase likelihood by 47.5%

exp(confint(m1))  # Confidence Intervals (by default 95%) for Regression Model Coefficients
#with 95% probability, the bundle increases the purchase likelihood around 28.2% to 69.8%

# Model 2:
m2 <- glm(IFUseCoupon ~ IFBundle + Channel, data = data, family = 'binomial')

summary(m2)  # deviance decreased and all the variables coefficients are significant base on Wald test

#why negative contribution of the promotion bundle?
exp(coef(m2))
exp(confint(m2))

#we include `Channel` variable to the previous model, why IFBundleBundle Coefficient is -0.56055 here? this model says us, if we offer bundle to people, this will reduce the likelihood of purchase
#how is it possible? Simpson's paradox: this happened because we didn't consider the impact of third variable which is very important and has impact on both X and Y at first of Analysis

# Model 3: model Interaction Effect
m3 <- glm(IFUseCoupon ~ IFBundle + Channel + IFBundle:Channel, data = data, family = 'binomial')

summary(m3)

#Interpret Coefficients of created model consider the base state of Categorical variables
 #base state (all variables be zero and just have Intercept): Mail + NoBundle -> log(Odds) = 0.25570
 #just IFBundleBundle = 1 with Intercept: Mail + Bundle -> log(Odds) = -0.87379 -> decrease purchase likelihood (IFUseCoupon probability)
  # compare -0.87379 with 0.25570 -> naturally, Channel can not impact on purchase likelihood directly because there can not be any causality between Channel and purchase likelihood. it doesn't make sense. but how this happened!?
  #                                  Channel does not cause of accepting bundle or reject it; in fact, Channel sets the customer types which receive our coupons. the target people of each Channel are not of same type!
  #                                  the target people of Email channel are young people -> is our bundle offer interesting for them?
  #                                  the target people of Mail channel are old people -> is our bundle offer interesting for them?
 #just ChannelInPerson = 1 with Intercept: InPerson + NoBundle -> log(Odds) = 1.50145 -> increase purchase likelihood (IFUseCoupon probability) -> InPerson is an effective Channel without considering bundle offer
 #just ChannelEmail = 1 with Intercept: Email + NoBundle -> log(Odds) = -3.14401 -> is it possible that our e-mails goes to Spam box of people?! and they didn't receive it, or open it!
 #ChannelEmail = 1 and IFBundleBundle = 1 and IFBundleBundle:ChannelEmail = 1 with Intercept: Email + Bundle -> log(Odds) = 2.98084
 #ChannelInPerson = 1 and IFBundleBundle = 1 and IFBundleBundle:ChannelInPerson = 1 with Intercept: InPerson + Bundle -> log(Odds) = 0.16937

#summary of deduction from data:
 #good reason to continue the bundling promotion campaign by email. but its success there does not necessarily imply success of in-person offers or through a regular mail campaign.
 # in other words: it seems that the people who receive bundle offer via Email channel, the bundle offer is interesting for them. but consider that InPerson channel is more efficient than Email channel.
 #                 and it seems that the people who receive bundle offer via Mail channel, the bundle offer is not interesting for them.

#Deviance chi-squared test
#H0: the model is not better than chance at predicting the outcome
#p-value < 0.05 reject H0
modelChi3 <- m3$null.deviance - m3$deviance  # test statistic
Chidf3 <- m3$df.null - m3$df.residual  # chi-squared degree of freedom
Chisq_prob3 <- 1 - pchisq(modelChi3, Chidf3)  # p-value
Chisq_prob3

#result: this model is better than base model
