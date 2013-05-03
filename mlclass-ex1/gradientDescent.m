function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	%h=X*theta;
	%delta=1/m*(sum((h-y)'*X));  %delta is only 1x1 dimension
	%delta=1/m*(sum(h-y)*sum(X)); %array correct but turns to NaN
	%delta=1/m*(h-y).*X;
	delta=1/m*(X'*X*theta-X'*y);
	theta=theta-alpha.*delta;
	%fprintf(theta);




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
J_history
end
