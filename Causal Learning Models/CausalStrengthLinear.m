% In this program, we use the linear likelihood function and the uniform 
% prior of causal strengths to implement causal inference with the sampling
% approach. We will compute the posterior distribution of causal strengths,
% and then estimate the median value of strength that a drug causes an 
% individual to recover from a cold, based on contingency data that report 
% the number of participants in each condition.

% The linear likelihood function is another approach used in probabilistic 
% graphical models to model causal strengths or dependencies between 
% variables. Unlike the noisy-or likelihood function, which assumes a 
% binary relationship between cause and effect, the linear likelihood 
% function assumes a linear relationship.

% The linear likelihood function assumes that the relationship between the 
% cause variables and the effect variable is additive and proportional. It 
% means that the effect variable is a sum of the weighted contributions 
% from the cause variables. The weights determine the magnitude and 
% direction of the influence of each cause variable on the effect variable.


clear all; close all;

% Input contingency data
a = 8;  % B = 1, C = 1, E = 1
b = 0;  % B = 1, C = 1, E = 0
c = 6;  % B = 1, C = 0, E = 1
d = 2;  % B = 1, C = 0, E = 0

% B: immune system working to recover from cold (or not)
% C: took the experimental drug (or not)
% E: recovered from a cold within three days (or not)

%% Step 1: Generate samples from prior distriution
samplenum = 100000;

w0 = unifrnd(0, 1, [1,samplenum]) ; % sample of causal strengths for w0 (weight of background cause)
w1 = unifrnd(0, 1, [1,samplenum]) ; % sample of causal strengths for w1 (weight of hypothesized cause)

%% Step 2: Compute likelihood using linear function

% enforce constraint that for any sample, w0 + w1 <= 1
wRemoveMask = w0 + w1 <= 1 ; % 1x100000 logical vec (1s where constraint is satisfied, 0 where it isn't)

w0constr = w0(wRemoveMask) ; % ~ 1x50000 vec: constrained w0
w1constr = w1(wRemoveMask) ; % ~ 1x50000 vec: constrained w1

% B = 1, C = 1, E = 1
likeli_a = ( w0constr + w1constr ).^a ; % linear likelihood function: w0(B) + w1(C) 

% B = 1, C = 1, E = 0
likeli_b = ( 1 - (w0constr + w1constr) ).^b ;

% B = 1, C = 0, E = 1
likeli_c = w0constr.^c ;      % likelihood of recovery in 3 days just due to immune system
                     
% B = 1, C = 0, E = 0
likeli_d = (1-w0constr).^d ;  % likelihood of no recovery in 3 days just due to immune system


likelihood = likeli_a .* likeli_b .* likeli_c .* likeli_d ; % ~ 1x50000 vec

%% Step 3: Normalize likelihood to compute the sampling weights
weight = likelihood / sum(likelihood) ;  % ~ 1x50000 vec

%% Step 4: Generate the posterior random samples
postindx = randsample(1:length(w0constr),samplenum,true,weight) ;
% use an index for the pool because we have 2 prior distributions 
% we can use this index to get to the individual posterior samples

%% Step 5: Plot the histograms of posterior samples for wb and wc
postwbsample = w0constr(postindx); % posterior samples of causal strenghts for B
subplot(1,2,1); 
hist(postwbsample,50); 
title('Wb hist');

postwcsample = w1constr(postindx); % posterior samples of causal strengths for C
subplot(1,2,2); 
hist(postwcsample,50);
title('Wc hist'); 

%% Step 6: Compute the median value of w1
disp('Median estimate of the causal strength of the drug (w1)')
median(postwcsample)
