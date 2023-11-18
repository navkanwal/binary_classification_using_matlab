function [J, grad] = costFunction(theta, X, y)



m = length(y); 


J = 0;
grad = zeros(size(theta));


h=sigmoid(X*theta);
Err=y.*log(h)+(1-y).*log(1-h);
J=-sum(Err)/m;

for j=1:1:length(theta)   
    Er=(h-y).*X(:,j);
    grad(j)=sum(Er)/m;
end



end
