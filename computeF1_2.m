function [F1 TP FP TN FN] = computeF1_2(Y, Y_hat)
%统计真阳性、真阴性、假阳性和假阴性的个数，计算F1分数
% Y: n*1
% Y_hat: n*1

% in case the format is not correct
Y(Y<=0) = -1;
Y(Y>0) = 1;
Y_hat(Y_hat<=0) = -1;
Y_hat(Y_hat>0) = 1;

% start
n = size(Y,1);
%返回相应的个数
TP = nnz(Y_hat==1&Y==1);
FP = nnz(Y_hat==1&Y==-1);
FN = nnz(Y_hat==-1&Y==1);
TN = nnz(Y_hat==-1&Y==-1);

if n~=(TP+FP+FN+TN)
    disp('n~=(TP+FP+FN+TN)!');
    F1 = 0;
else
    %计算精确率
    if TP+FP == 0
        prec = 0;
    else
        prec = TP/(TP+FP);
    end
    %计算召回率
    if TP + FN == 0
        rec = 0.5;
    else
        rec = TP/(TP+FN);
    end
    %计算F1分数
    if prec+rec == 0
        F1 = 0;
    else
        F1 = 2*prec*rec / (prec+rec);
    end
    %fprintf(1, 'F1 = %d ---- TP:%d, FP:%d, FN:%d, TN:%d, Prec:%d, rec:%d\n', F1, TP, FP, FN, TN, prec, rec);
end