function [F1] = computeF1_3(TP, FP, TN, FN)
%����F1��

%���㾫ȷ��
if TP+FP == 0
    prec = 0;
else
    prec = TP/(TP+FP);
end
%�����ٻ���
if TP + FN == 0
    rec = 0.5;
else
    rec = TP/(TP+FN);
end
%����F1����
if prec+rec == 0
    F1 = 0;
else
    F1 = 2*prec*rec / (prec+rec);
end
    