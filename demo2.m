function [STOP1,STOP2] = demo2(X,Y,fStr)

nFea=size(X,2);
nLabel = size(Y,2);
k_knn=50; %Take the number of neighbors
nRun=2;
%lambda1=0.1,lambda2=0.1 cal500 mu=6*1e4
lambda1 = 0.1;
lambda2 = 0.3;
gama = 0.01;

%Performance measure
perf_measure = @computeF1;
perf_measure2 = @computeF1_2;

%Number of output code components used
%but in this code, we also try nComps = 0,...,nLabel
nComps = [1:nLabel];

%result
macroF1 = zeros(nRun, length(nComps));
macroF1_labels = zeros(nRun, length(nComps), nLabel);
microF1 = zeros(nRun, length(nComps));
ZeroOneLoss = zeros(nRun, length(nComps));
exampleF1 = zeros(nRun,length(nComps));
exampleF1_labels = zeros(nRun, length(nComps), nLabel);
Hamming = zeros(nRun,length(nComps));

[m,n]=size(X);
s = RandStream('mt19937ar','Seed',1);  % set random seed make the experiment can repeat
RandStream.setGlobalStream(s);
indices = crossvalind('Kfold',m,nRun);  %Generate nRun folds, that is, there are equal proportions of 1-nRun in the indicators

for r = 1 : nRun
    test = (indices == r);  %Logic judgment, select a fold as the test set each time
    train = ~test;   %The remaining nRun-1 folds are the training set
    training_dataset_x=X(train,:);
    tsX=X(test,:);
    training_dataset_y=Y(train,:);
    tsY=Y(test,:);
%     training_dataset_x=imnoise(training_dataset_x,'salt & pepper',0.1)
    nTr = size(training_dataset_x,1);
    nTs = size(tsX,1);

    tic;
   [P,eigV,STOP1,STOP2] = Rfeature_extract(training_dataset_x,training_dataset_y,lambda1,lambda2);
   [P,eigV,STOP1,STOP2] = Rfeature_extract_KNN(training_dataset_x,training_dataset_y,lambda1,lambda2,gama,k_knn);
    toc;
  
    v = eigV; 
    [U,S,V]=svd(v); %SVD
    tic; 
    %start decoding and evaluation
    %iterate over the number of output projections used
    % 接下来：主要的功能是解码和评估（CCA那篇论文有讲到解码）
    % 第一层循环：选择从V的10维开始循环（比如：V依次在10、11、12......维的情况下分类的评估效果）-特征选择的效果
    for k = 2: size(v,2)
        % 这边就是根据循环的k值得到V在k维的矩阵（因为使用的是nuclear norm 所以这里用到分解中的U）--投影矩阵
        CMtx = U(:,1:k);%Projection matrix
        tsY_hat = zeros(size(tsX,1),nLabel);%nTs*nLabel
        % 公式||Vt(Y-PtX)||=||VtY->VtPtX||
        % 这边求的是：VtPt
        vp=CMtx'*P';
        % 这边求的是：VtPtX 这里是针对训练集中的X求解
        hb=vp*training_dataset_x';  % for evey test sample, it is not necessary to compute vp*training_dataset_x(j,:)' every time, it cost too much. we can compute it in the outside of the loop                                 
        % 循环测试集数据的数量
        for i = 1 : nTs    % for every test sample i in tsX
            % 这边求的是每一条测试数据乘以VP
            vp_tes=vp*tsX(i,:)'; %V'*M' projection for each sample in the test set
            % 旁边有注释解释
            hc=repmat(vp_tes,1,nTr); %Generate 1 row of nTr column subspaces, then stack the contents of vp_tes into matrix hc to generate a matrix of size(vp_tes, 1)*vp_tes
            % 计算测试集中每个样本与训练集中所有样本之间的距离
            Encoding_distance=(diag(S(1:k,:)))' * (hc - hb).^2; %Calculate the distance between each sample in the test set and all samples in the training set
            % 对与所有样本的距离进行排序
            [sA,index]=sort(Encoding_distance); % 1 * nTs-1 sort form small to big  SA: 1*nTs-1 index: 1*nTs-1
            % 获取非零元素
            none_zone=find(sA); 
            % 舍去一列都为0的数据                         
            tem_ma=~max(training_dataset_y,[],1)==0; % union of label of 0
            % 取最小的前K个距离的倒数
            tempt=1./sA(none_zone(1:k_knn)); %Take the reciprocal of the smallest first k distance
            % 根据选取前几个训练的Y值的计算预测出Y值（这里面具体的机制我也没有钻研透，师姐说就是在CCA和2012那篇copy过来的）
            tempcomp=tempt*training_dataset_y(index(none_zone(1:k_knn)),tem_ma)./sum(tempt);
            % 重排测试集第i个样本的标签输出（大于0.5的为1，小于0.5为0）
            tsY_hat(i,tem_ma)=tempcomp>=0.5; %Predict the label output of the i-th sample of the test set    
        end
        %预测的时候Y怎么用的
        % 上面循环获得重新修改好的tsY_hat为预测的Y值
        % 接下来就是把预测出来的tsY_hat和测试Y值进行比较，获得评估数据
        %evaluation，包括四方面
        %measure the macroF1
        for t = 1 : nLabel
            macroF1_labels(r,k,t) = perf_measure2(tsY(:,t), tsY_hat(:,t)); %计算每个标签的F1值
        end
        macroF1(r,k) = mean(macroF1_labels(r,k,:)); %求平均值
        
        %also measure the microF1
        TP = zeros(3,nLabel); FP = zeros(3,nLabel); TN = zeros(3,nLabel); FN = zeros(3,nLabel);
        for t = 1 : nLabel
            [tempF,TP(1,t),FP(1,t),TN(1,t),FN(1,t)] = computeF1_2(tsY(:,t), tsY_hat(:,t)); %计算每个标签的TP、FP、TN、PN值
        end       
        microF1(r,k) = computeF1_3(sum(TP(1,:)), sum(FP(1,:)), sum(TN(1,:)), sum(FN(1,:)) );
        
        %also measure the subset accuracy
        for i = 1 : nTs
            ZeroOneLoss(r,k) = ZeroOneLoss(r,k) + ( nnz(tsY(i,:)~=tsY_hat(i,:))~=0 );
        end
        ZeroOneLoss(r,k) = ZeroOneLoss(r,k)./nTs;
        
        %measure the eampleF1 performance 计算评价指标exampleF1
        for t = 1 : nTs
            exampleF1_labels(r,k,t) = perf_measure2(tsY(t,:)', tsY_hat(t,:)');                
        end
        exampleF1(r,k) = mean(exampleF1_labels(r,k,:));
        
        %measure hamming loss          
        Hamming(r,k) = sum(sum(tsY ~= tsY_hat))/(nTs*nLabel);
    end
    toc;
end
SubAccuracy = 1 - ZeroOneLoss;

%求mean和standard error
mea_Hamm=roundn(mean(Hamming),-4);% 求平均值
st_Hamm=roundn(std(Hamming),-4); % std为求Hamming的标准方差，roundn是四舍五入到小数点右边第四位
mea_Sub=roundn(mean(SubAccuracy),-4);
st_Sub=roundn(std(SubAccuracy),-4);
mea_Mac=roundn(mean(macroF1),-4);
st_Mac=roundn(std(macroF1),-4);
mea_Mic=roundn(mean(microF1),-4);
st_Mic=roundn(std(microF1),-4);
mea_exam=roundn(mean(exampleF1),-4);
st_exam=roundn(std(exampleF1),-4);
%可视化显示
%绘图
figure();
plot(mea_exam,'LineWidth',1);
legend('exampleF1','HammingLoss');
%输出格式
save(sprintf('New_%s',fStr), 'mea_exam','st_exam','mea_Hamm','st_Hamm', 'mea_Sub','st_Sub','mea_Mac','st_Mac','mea_Mic','st_Mic','microF1', 'SubAccuracy','macroF1', 'nComps','Hamming','exampleF1');

