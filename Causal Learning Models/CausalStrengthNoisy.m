% In this program, we use the noisy-or likelihood function and the uniform 
% prior of causal strengths to implement causal inference with the sampling
% approach. We will compute the posterior distribution of causal strengths,
% and then estimate the median value of strength that a drug causes a 
% headache, based on contingency data that report the number of 
% participants in each condition.

% B: background causes of headaches (present in all groups)
% C: took the drug (1) or did not take the drug (0)
% E: had a headache (1) or did not have a headache (0)

% The noisy-or likelihood function is commonly used in probabilistic 
% graphical models to model causal strengths or dependencies between 
% variables. It is particularly useful when dealing with binary variables 
% and capturing the idea that a cause may not always lead to an effect due 
% to noise or other unobserved factors.

% In the context of a causal graph, the noisy-or likelihood function 
% represents the probability of an effect variable being true given a set 
% of cause variables. It assumes that each cause variable has a certain 
% "causal strength" or influence on the effect variable, and the effect 
% variable will be true if and only if at least one of the cause variables 
% is true.

% The noisy-or assumption allows for a flexible representation of causal 
% dependencies, where multiple causes can contribute to an effect but not 
% all causes are necessary for the effect to occur. The noise factor 
% accounts for the possibility of false negatives or situations where a 
% cause does not lead to the effect due to noise or other unobserved 
% factors, which isn't always the case in animal reasoning (rats and
% pigeons, for example, seem to rely solely on a linear likelihood
% function, where the presence of causes and effects are simply summed in
% accordance fundamentally with associative learning).

% Note: The noisy-or likelihood function assumes that the cause variables 
% are conditionally independent given the effect variable. 

clear all;

% Input contingency data
a = 12;  % B = 1, C = 1, E = 1
b = 4;   % B = 1, C = 1, E = 0
c = 0;   % B = 1, C = 0, E = 1
d = 16;  % B = 1, C = 0, E = 0


%% Step 1: general samples from prior distriution
samplenum = 100000;

w0 = unifrnd(0, 1, [1,samplenum]) ;
w1 = unifrnd(0, 1, [1,samplenum]) ;

%% Step 2: compute likelihood using noisy-or function
% B = 1, C = 1, E = 1
likeli_a = ( 1 - (1 - w0) .* (1 - w1) ).^a ;

% B = 1, C = 1, E = 0
likeli_b = ( (1 - w0) .* (1 - w1) ).^b ;

% B = 1, C = 0, E = 1
likeli_c = w0.^c ;

% B = 1, C = 0, E = 0
likeli_d = (1-w0).^d ;


likelihood = likeli_a .* likeli_b .* likeli_c .* likeli_d ;

%% Step 3: Normalize likelihood to compute the sampling weights
weight = likelihood / sum(likelihood) ;

%% Step 4: generate the posterior random samples
postindx = randsample(1:samplenum,samplenum,true,weight) ;
% use an index for the pool because we have 2 prior distributions 
% we can use this index to get to the individual posterior samples

%% Step 5: plot the histograms of posterior samples for wb and wc
postwbsample = w0(postindx);
subplot(1,2,1); 
hist(postwbsample,50); 
title('Wb hist');

postwcsample = w1(postindx);
subplot(1,2,2); 
hist(postwcsample,50);
title('Wc hist'); 


%% Step 6: Compute the median value of w1
disp('Estimate of the causal strength')
median(postwcsample)
