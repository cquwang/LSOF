function [lsof_vector,outliers,auc] = lsof(ds,epsilon)
% Local structure outlier factor
%   Input:
%   ds: original dataset
%   epsilon: neighborhood size of local structure
%   Output:
%   lsof_vector: outlier factor of each local structure
%   outliers: detection result
%   auc: AUC value

%%initializing data
data = ds(:,1:end-1); % attributes
answer = ds(:,end); % labels
[r,~] = size(answer);
%% k-nearest neighbors and indecies of each data point
[k_dist,k_index] = pdist2(data,data,'euclidean','Smallest',epsilon+1);
k_dismatrix = k_dist(2:end,:);

%% construct local structures
[local_structure_matrix,local_structure_matrix_dis] = construct_local_structure_matrix(data,epsilon,k_dismatrix,k_index);
%show local structure graph
figure
scatter(data(:,1),data(:,2),40,'.');
hold on
xlabel('x');
ylabel('y');
title('LNNC graph');
plotnns(data,local_structure_matrix);
hold off

%% compute LSOF values
lsof_vector = compute_lsof(local_structure_matrix_dis);

[sorted_lsof,~] = sort(lsof_vector,'descend');
% show sorted lsof values
figure
plot(sorted_lsof,'.-');
hold on
xlabel('No. of LNNC');
ylabel('LSOF');
% choose top-n
[~,y,~] = ginput(1);
[outlier_lnnc_numbers,~,~] = find(lsof_vector >= y);
topn = length(outlier_lnnc_numbers);
title(strcat('topn = ',num2str(topn)));
hold off
% show outlier local structures
outlier_lnncs = local_structure_matrix(outlier_lnnc_numbers,:);
figure
scatter(data(:,1),data(:,2),40,'.');
hold on
plotnns(data,outlier_lnncs);
hold off
% separate outliers from outlier-LNNCs
outliers = separate_outliers(outlier_lnnc_numbers,local_structure_matrix,local_structure_matrix_dis);
% show result
figure
scatter(data(:,1),data(:,2),40,'.');
hold on;
xlabel('x');
ylabel('y');
scatter(data(outliers,1),data(outliers,2),'r*');
hold off

%%compute AUC
    scores = zeros(r,1);
    scores(outliers) = 1; % outlier scores
    proclass = 1; 
    
    [~,~,~,auc] = perfcurve(answer,scores,proclass);
 
end

