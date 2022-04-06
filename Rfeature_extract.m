function[P,eigV,STOP1,STOP2] = Rfeature_extract(dataX,dataY,lambda1,lambda2)

%此时dataX，dataY都以行向量存放
n = size(dataX,1);
p = size(dataX,2);
q = size(dataY,2);

%Transpose row-by-row discharge into column-by-row discharge
dataX = dataX';%X(pxn)
dataY = dataY';%Y(qxn)

%Start main loop
tol = 1e-7;
maxIter = 1e4;
iter = 0;
rho = 1.1;
%mu=6*1e4 lambda1=43
mu=1e4;
max_mu = 1e14;

%Initializaing
P = zeros(p,q);
eigV = zeros(q,q);
Y1 = zeros(p,q);
Y2 = zeros(q,q);
I1 = eye(q,q);
while iter < maxIter
    iter = iter + 1;
    % update J
    temp = P + Y1/mu;
    [U,sigma,V] = svd(temp,'econ');
    sigma = diag(sigma);
    svp = length(find(sigma>lambda1/mu));
    if svp >= 1
        sigma = sigma(1:svp)-lambda1/mu;
    else
        svp = 1;
        sigma = 0;
    end
    J = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
    % update P
    %P = (1/2)*(J+Q-(1/mu)*(-2*dataX*dataY'*eigV*eigV'+dataX*dataX'*Q*eigV*eigV'+Y1+Y3));
    A = mu*pinv(dataX*dataX'+eps)+eps*eye(p,p);
    B = 2*eigV*eigV'+eps*eye(q,q);
    C = pinv(dataX*dataX'+eps)*(2*dataX*dataY'*eigV*eigV'-Y1+mu*J+eps)+eps*eye(p,q);
    P = sylvester(A,B,C);
    % update W
    temp = eigV + Y2/mu;
    [U,sigma,V]=svd(temp,'econ');
    sigma = diag(sigma);
    svp = length(find(sigma>lambda2/mu));
    if svp >= 1
        sigma = sigma(1:svp);
    else
        svp = 1;
        sigma = 0;
    end
    W = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
    % update eigV
    if ~any(eigV(:))
        eigV = pinv(2/mu*(dataY*dataY'-2*P'*dataX*dataY'+P'*dataX*dataX'*P)+I1)*(W-Y2/mu+eye(q,q));
    else
        eigV = pinv(2/mu*(dataY*dataY'-2*P'*dataX*dataY'+P'*dataX*dataX'*P)+I1)*(W-Y2/mu);
    end
    leq1 = P-J;
    leq2 = eigV-W;
    STOP1(iter) = max(max(abs(leq1)));
    STOP2(iter) = max(max(abs(leq2)));
 
    stopA = max(max(max(abs(leq1)),max(max(abs(leq2)))));
    if stopA <tol
        break;
    else
        Y1 = Y1+mu*leq1;
        Y2 = Y2+mu*leq2;
        mu = min(max_mu,mu*rho);
    end
    disp([iter]);
end

