function [F1] = computeF1_3(TP, FP, TN, FN)
%计算F1分

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
    