function g = sigmoid(z)

g = zeros(size(z));
[a,b]=size(z);

for i=1:1:a
    for j=1:1:b
g(i,j)=1/(1+exp(-z(i,j)));
    end
end



end
