function p = predict(theta, X)

m = size(X, 1); 


p = zeros(m, 1);


h=sigmoid(theta'*X');

for i=1:1:length(h)
    if (h(i) >=0.5)
        p(i)=1;
    else
        p(i)=0;
    end
end
end
