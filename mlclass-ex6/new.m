n = 500;

% You need to return the following variables correctly.
x = zeros(n, 1);

word_indices=[100,212,312,423,345,46,37];

for i=1:length(word_indices)
	if (word_indices(i))
		x(i)=1
	else
		x(i)=0;
	end
end