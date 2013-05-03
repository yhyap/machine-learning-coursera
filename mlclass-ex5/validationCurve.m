function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector lambda_vec contains the different lambda parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               you the errors obtained after training with 
%               lambda = lambda_vec(i)
%
% Note: You can loop over lambda_vec with the following:
%
%       for i = 1:length(lambda_vec)
%           lambda = lambda_vec(i);
%           % Compute train / val errors when training linear 
%           % regression with regularization parameter lambda
%           % You should store the result in error_train(i)
%           % and error_val(i)
%           ....
%           
%       end
%
%
for i=1:length(lambda_vec)
	lambda=lambda_vec(i);
	[theta]=trainLinearReg(X, y, lambda);
	lambda=0;	% Set to zero when calculating error since already been trained
	% We are calc J as error not as cost. So, lambda should not be included when 
	% calculating error for thetas that have been trained, else it will be biased 
	% Refer to 2.1 of Exercise 5, error is computed without lambda
	[e_train]=linearRegCostFunction(X, y, theta, lambda);
	[e_val]=linearRegCostFunction(Xval, yval, theta, lambda);	% J over all CV set for new set of theta

% Accumulating error from i=1:m
	if (i==1)
		error_train=e_train;
		error_val=e_val;
	else
		error_train=[error_train; e_train];
		error_val=[error_val; e_val];
	end
end

% Above methods are also accepted when submitted, following method are also accepted

%for i=1:length(lambda_vec)
%lambda=lambda_vec(i);
%[theta] = trainLinearReg(X, y, lambda);

% For training set
%m=length(y);
%thetas=theta(2:end,1);
%h=X*theta;
%lambda=0;
%error_train(i)=(1/(2.*m))*sum((h-y).^2)+(lambda/(2.*m)).*sum(thetas.^2);

% For CV set
%n=length(yval);
%h2=Xval*theta;
%lambda=0;
%error_val(i)=(1/(2.*n))*sum((h2-yval).^2)+(lambda/(2.*n)).*sum(thetas.^2);
%end

% =========================================================================

end
