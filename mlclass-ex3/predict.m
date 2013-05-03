function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%%% Map from Layer 1 to Layer 2
% Coverts to matrix of 5000 examples x 26 thetas
z1=X*Theta1';
% Sigmoid function converts to p between 0 to 1
h1=sigmoid(z1);

%%% Map from Layer to Layer 3
% Add ones to the h1 data matrix
h1=[ones(m, 1) h1];
% Converts to matrix of 5000 exampls x num_labels 
z2=h1*Theta2';
% Sigmoid function converts to p between 0 to 1
h2=sigmoid(z2);

% pval returns the highest value in each row, while p returns the position in each row
[pval, p]=max(h2,[],2);  







% =========================================================================


end
