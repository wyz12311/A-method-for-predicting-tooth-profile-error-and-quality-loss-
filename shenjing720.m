clc
clear all
close all
% 开始计时
tic;
% 读取数据
dataO1 = xlsread('C:\Users\Administrator.DESKTOP-JC66FHP\Desktop\date.xlsx');

% 定义输入特征和输出特征的数量
num_input_features = 18;
num_output_features = 6;

% 选择前18列作为输入特征，后两列作为输出特征
x_feature_label = dataO1(:, 1:num_input_features);
y_feature_label = dataO1(:, num_input_features + 1:num_input_features + num_output_features);

% 划分训练集和预测集
train_data = dataO1(1:480, :);
test_data = dataO1(481:500, :);

% 提取训练集和测试集的输入和输出特征
train_x_feature_label = train_data(:, 1:num_input_features);
train_y_feature_label = train_data(:, num_input_features + 1:num_input_features + num_output_features);

test_x_feature_label = test_data(:, 1:num_input_features);
test_y_feature_label = test_data(:, num_input_features + 1:num_input_features + num_output_features);

% 标准化训练集和测试集的输入和输出特征
x_mu = mean(train_x_feature_label);
x_sig = std(train_x_feature_label);
train_x_feature_label_norm = (train_x_feature_label - x_mu) ./ x_sig;
test_x_feature_label_norm = (test_x_feature_label - x_mu) ./ x_sig;

y_mu = mean(train_y_feature_label);
y_sig = std(train_y_feature_label);
train_y_feature_label_norm = (train_y_feature_label - y_mu) ./ y_sig;

% 构建神经网络模型
% 两个隐藏层，每层各10个神经元
hidden_layer_sizes = [10, 10];  % 隐藏层大小的数组
Mdl = newff(train_x_feature_label_norm', train_y_feature_label_norm', hidden_layer_sizes, {'logsig', 'logsig', 'purelin'});

% 训练神经网络模型
[Mdl, ~] = train(Mdl, train_x_feature_label_norm', train_y_feature_label_norm');

% 使用训练好的模型进行预测
y_test_predict_norm = sim(Mdl, test_x_feature_label_norm')';
y_test_predict = y_test_predict_norm .* y_sig + y_mu;

% % 绘制 y1 的预测值和实际值比较图
% subplot(2, 1, 1);  % 分为两行一列的图形的第一行
% plot(test_y_feature_label(:,1), 'o-', 'LineWidth', 2, 'MarkerSize', 6);  % 实际值
% hold on;
% plot(y_test_predict(:,1), 'x--', 'LineWidth', 2, 'MarkerSize', 6);  % 预测值
% title('Comparison of Actual and Predicted Y1');
% xlabel('Sample Index');
% ylabel('Y1 Value');
% legend('Actual Y1', 'Predicted Y1');
% grid on;
% hold off;
% 
% % 绘制 y2 的预测值和实际值比较图
% subplot(2, 1, 2);  % 分为两行一列的图形的第二行
% plot(test_y_feature_label(:,2), 'o-', 'LineWidth', 2, 'MarkerSize', 6);  % 实际值
% hold on;
% plot(y_test_predict(:,2), 'x--', 'LineWidth', 2, 'MarkerSize', 6);  % 预测值
% title('Comparison of Actual and Predicted Y2');
% xlabel('Sample Index');
% ylabel('Y2 Value');
% legend('Actual Y2', 'Predicted Y2');
% grid on;
% hold off;

% % 显示图形
% figure(gcf);

%%
rng(123);  % 设定随机种子以获得可重复的结果

% 参数个数(前18个为变量)
nParameters = 18;

% % 参数上下界
% param_lb =[-0.001, -0.001, -0.001, -0.001, -0.001,  -0.001, -0.002,-0.010, -0.005, 0, 0, 0, 0, 0, 0, 0, 0, 0]; % 参数下界
% param_ub =[0.001, 0.001, 0.001, 0.001,0.001,0.001, 0.002, 0.010, 0.006, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1, 1, 1]; % 参数上界
% 参数上下界
param_lb =[-0.001, -0.001, -0.001, -0.001, -0.001,  -0.001, 0,-0.0011, -0.0044,0, 0, 0, 0, 0, 0, 0, 0, 0]; % 参数下界
param_ub =[0.001, 0.001, 0.001, 0.001,0.001,0.001, 0.0012, 0.0026, -0.002, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1, 1, 1]; % 参数上界

% 设定MCMC参数
nIterations1 = 500; % 先验样本的迭代次数
nBurnIn1 = 100; % 先验样本的燃烧期
nChains1 = 500; % 先验样本的链数

nIterations2 =500; % 后验样本的迭代次数
nBurnIn2 = 100; % 后验样本的燃烧期
nChains2 = 500; % 后验样本的链数

% 初始化MCMC链的起始值（在上下界范围内随机初始化）
initial_values = bsxfun(@plus, param_lb, bsxfun(@times, param_ub - param_lb, rand(nChains1 + nChains2, nParameters)));
% 存储MCMC链的样本
chain_samples1 = NaN(nIterations1, nChains1, nParameters);
chain_samples2 = NaN(nIterations2, nChains2, nParameters);
% 存储后验分布样本
posterior_samples = NaN(nChains2, nParameters);

% 存储先验样本的目标函数值
model_outputs_prior = NaN(nChains1, 2, nIterations1);  % 2 表示目标函数有两个输出

% 估算均值和标准差
param_mean = (param_lb + param_ub) / 2; 
param_std = (param_ub - param_lb) / (2 * sqrt(3)); 

% 存储均值和标准差
for i = 1:length(param_mean)
    eval(['mu_param', num2str(i), ' = param_mean(i);']);
    eval(['sigma_param', num2str(i), ' = param_std(i);']);
end

% 存储均值和标准差
mu_prior = [mu_param1, mu_param2, mu_param3, mu_param4, mu_param5, mu_param6, mu_param7, mu_param8, mu_param9, mu_param10, mu_param11, mu_param12, mu_param13, mu_param14, mu_param15, mu_param16, mu_param17, mu_param18];
sigma_prior = [sigma_param1, sigma_param2, sigma_param3, sigma_param4, sigma_param5, sigma_param6, sigma_param7, sigma_param8, sigma_param9, sigma_param10, sigma_param11, sigma_param12, sigma_param13, sigma_param14, sigma_param15, sigma_param16, sigma_param17, sigma_param18];

% 先验样本生成
for chain1 = 1:nChains1
    current_params1 = initial_values(chain1, :);
    accepted_samples1 = zeros(nIterations1- nBurnIn1, nParameters);
    count1 = 1;
    for iteration1 = 1:nIterations1
        proposed_params1 = current_params1;
        % 正态分布采样更新参数
        proposed_params1(1:9) = normrnd(mu_prior(1:9), sigma_prior(1:9)); 
        proposed_params1(10:18) = normrnd(mu_prior(10:18), sigma_prior(10:18)); 
        
        % 模型预测
        y_current_norm1 = sim(Mdl, proposed_params1')';
        Y_current1 = y_current_norm1 .* y_sig + y_mu; % 反归一化

        % 应用约束条件
        if Y_current1(1) == 0
            Y_current1(2) = 0;
        else
            Y_current1(2) = max(0, Y_current1(2));
        end

        % 存储模型输出
        model_outputs_prior(iteration1, 1, chain1) = Y_current1(:, 1);
        model_outputs_prior(iteration1, 2, chain1) = Y_current1(:, 2);

        % 更新参数
        current_params1 = proposed_params1;
        if iteration1 > nBurnIn1
            accepted_samples1(count1, :) = proposed_params1;
            count1 = count1 + 1;
        end
    end
    chain_samples1(1:nIterations1 - nBurnIn1, chain1, :) = accepted_samples1;
end

% % 先验样本生成
% for chain1 = 1:nChains1
%     current_params1 = initial_values(chain1, :);
%     accepted_samples1 = zeros(nIterations1 - nBurnIn1, nParameters);
%     count1 = 1;
%     for iteration1 = 1:nIterations1
%         proposed_params1 = current_params1;
%         % 正态分布采样更新参数
%         proposed_params1(1:9) = normrnd(mu_prior(1:9), sigma_prior(1:9)); 
%         proposed_params1(10:18) = normrnd(mu_prior(10:18), sigma_prior(10:18)); 
%         
%         % 模型预测
%         y_current_norm1 = sim(Mdl, proposed_params1')';
%         Y_current1 = y_current_norm1 .* y_sig + y_mu; % 反归一化
% 
%         % 应用约束条件
%         if Y_current1(1) == 0
%             Y_current1(2) = 0;
%         else
%             Y_current1(2) = max(0, Y_current1(2));
%         end
% 
%         % 存储模型输出
%         model_outputs_prior(iteration1, 1, chain1) = Y_current1(:, 1);
%         model_outputs_prior(iteration1, 2, chain1) = Y_current1(:, 2);
% 
%         % 更新参数
%         % 只在y2大于0时才更新和存储样本
%         if Y_current1(2) > 0
%             current_params1 = proposed_params1;
%             if iteration1 > nBurnIn1
%                 accepted_samples1(count1, :) = proposed_params1;
%                 count1 = count1 + 1;
%             end
%         end
%     end
%     % 存储每个链的采样结果，仅针对非燃烧期的迭代
%     chain_samples1(1:count1-1, chain1, :) = accepted_samples1(1:count1-1, :);
% end

% 
% 导入观测数据
% actual_data = xlsread('14.xls');
actual_data = xlsread('13.xls');
actual_Y1 = actual_data(1:148, 1);
actual_Y2 = actual_data(1:148, 2);

% 计算actual_data中y1和y2的均值
mean_actual_Y1 = mean(actual_data(1:148, 1));
mean_actual_Y2 = mean(actual_data(1:148, 2));

% 后验样本生成
model_outputs_posterior = NaN(nIterations2, 2, nChains2); % 存储后验样本的目标函

% 假设已经对 actual_Y1 进行了KDE，并获取了评估点和密度值
[f_Y1, xi_Y1] = ksdensity(actual_Y1);

% 假设 param_lb 和 param_ub 分别是参数的下界和上界
param_range = param_ub - param_lb;

% 初始尝试使用参数范围的一个小比例作为步长
step_size = 0.1 * param_range;

for chain2 = 1:nChains2
    current_params2 = initial_values(nChains1 + chain2, :);
    accepted_samples2 = zeros(nIterations2 - nBurnIn2, nParameters);
     count2 = 1;
    for iteration = 1:nIterations2
        % 使用神经网络模型进行当前参数的预测
        y_current_norm = sim(Mdl, current_params2')';
        Y_current = y_current_norm .* y_sig + y_mu; % 反归一化
        % 应用约束条件
        if Y_current(1) == 0
            Y_current(2) = 0;
        else
            Y_current(2) = max(0, Y_current(2));
        end
        % 计算当前状态基于y1的似然值
        likelihood_current_Y1 = interp1(xi_Y1, f_Y1, Y_current(:, 1), 'linear', 0);
        likelihood_current = likelihood_current_Y1;
             
        % 生成新的提议参数，通过在当前参数的基础上加入随机扰动
        proposed_params2 = current_params2 + normrnd(0, step_size, [1, nParameters]);
           
        % 对提议的参数生成模型预测
        y_proposed_norm = sim(Mdl, proposed_params2')';
        Y_proposed = y_proposed_norm .* y_sig + y_mu; % 反归一化
        % 应用约束条件
        if Y_proposed(1) == 0
            Y_proposed(2) = 0;
        else
            Y_proposed(2) = max(0, Y_proposed(2));
        end
        
        % 计算提议状态基于y1的似然值
        likelihood_proposed_Y1 = interp1(xi_Y1, f_Y1, Y_proposed(:, 1), 'linear', 0);
        likelihood_proposed = likelihood_proposed_Y1;
       
        % 将y1和y2的模型输出存储到model_outputs_posterior中
        model_outputs_posterior(iteration, 1, chain2) = Y_proposed(:, 1);
        model_outputs_posterior(iteration, 2, chain2) = Y_proposed(:, 2);
        
        % 计算先验概率，注意这里使用proposed_params2
        prior_current = prod(normpdf(current_params1, mu_prior, sigma_prior));
        prior_proposed = prod(normpdf(proposed_params1, mu_prior, sigma_prior));
        
        % 计算接受率
        acceptance_ratio = (likelihood_proposed * prior_proposed) / (likelihood_current * prior_current);

        if rand() < acceptance_ratio
            current_params2 = proposed_params2;
        end
        
        if iteration > nBurnIn2
            accepted_samples2(count2, :) = current_params2;
            count2 = count2 + 1;
        end
    end

% 存储每个链的采样结果，仅针对非燃烧期的迭代
chain_samples2(1:nIterations2 - nBurnIn2, chain2, :) = accepted_samples2;

end

% for chain2 = 1:nChains2
%     current_params2 = initial_values(nChains1 + chain2, :);
%     accepted_samples2 = zeros(nIterations2 - nBurnIn2, nParameters);
%     count2 = 1;
%     for iteration2 = 1:nIterations2
%         % 使用神经网络模型进行当前参数的预测
%         y_current_norm = sim(Mdl, current_params2')';
%         Y_current = y_current_norm .* y_sig + y_mu; % 反归一化
%         % 应用约束条件
%         if Y_current(1) == 0
%             Y_current(2) = 0;
%         else
%             Y_current(2) = max(0, Y_current(2));
%         end
%         % 计算当前状态基于y1的似然值
%         likelihood_current_Y1 = interp1(xi_Y1, f_Y1, Y_current(:, 1), 'linear', 0);
%         likelihood_current = likelihood_current_Y1;
%              
%         % 生成新的提议参数，通过在当前参数的基础上加入随机扰动
%         proposed_params2 = current_params2 + normrnd(0, step_size, [1, nParameters]);
%            
%         % 对提议的参数生成模型预测
%         y_proposed_norm = sim(Mdl, proposed_params2')';
%         Y_proposed = y_proposed_norm .* y_sig + y_mu; % 反归一化
%         % 应用约束条件
%         if Y_proposed(1) == 0
%             Y_proposed(2) = 0;
%         else
%             Y_proposed(2) = max(0, Y_proposed(2));
%         end
%         
%         % 计算提议状态基于y1的似然值
%         likelihood_proposed_Y1 = interp1(xi_Y1, f_Y1, Y_proposed(:, 1), 'linear', 0);
%         likelihood_proposed = likelihood_proposed_Y1;
%        
%         % 将y1和y2的模型输出存储到model_outputs_posterior中
%         if Y_proposed(2) > 0
%             model_outputs_posterior(iteration2, 1, chain2) = Y_proposed(:, 1);
%             model_outputs_posterior(iteration2, 2, chain2) = Y_proposed(:, 2);
%         end
%         
%         % 计算先验概率，注意这里使用proposed_params2
%         prior_current = prod(normpdf(current_params1, mu_prior, sigma_prior));
%         prior_proposed = prod(normpdf(proposed_params1, mu_prior, sigma_prior));
%         
%         % 计算接受率
%         acceptance_ratio = (likelihood_proposed * prior_proposed) / (likelihood_current * prior_current);
% 
%         if rand() < acceptance_ratio
%             current_params2 = proposed_params2;
%         end
%         
%         if iteration2 > nBurnIn2
%             % 只在y2大于0时才保存样本
%             if Y_proposed(2) > 0
%                 accepted_samples2(count2, :) = current_params2;
%                 count2 = count2 + 1;
%             end
%         end
%     end
% 
%     % 存储每个链的采样结果，仅针对非燃烧期的迭代
%     chain_samples2(1:nIterations2 - nBurnIn2, chain2, :) = accepted_samples2;
% 
% end

% 分析MCMC结果
mean_params_prior = squeeze(mean(chain_samples1, 1));
std_params_prior = squeeze(std(chain_samples1, 1));

mean_params_posterior = squeeze(mean(chain_samples2, 1));
std_params_posterior = squeeze(std(chain_samples2, 1));

% 提取y1和y2的后验样本值
Y1_values = reshape(model_outputs_posterior(:, 1, :), [], 1); % 将所有链的y1值提取到一个向量中
Y2_values = reshape(model_outputs_posterior(:, 2, :), [], 1); % 将所有链的y2值提取到一个向量中

% 筛选出y2 > 0的样本
valid_indices = Y2_values > 0;
Y1_values_filtered = Y1_values(valid_indices);
Y2_values_filtered = Y2_values(valid_indices);

%%
%y1热图
% 创建索引
x_values = (1:length(Y1_values_filtered))';
% 定义分箱数
num_bins_x = 20; % 样本索引的分箱数
num_bins_y = 20; % 数据值的分箱数
% 创建分箱边界
x_edges = linspace(min(x_values), max(x_values), num_bins_x + 1);
y_edges = linspace(min(Y1_values_filtered), max(Y1_values_filtered), num_bins_y + 1);
% 计算二维直方图
[counts_2d, ~, ~] = histcounts2(x_values, Y1_values_filtered, x_edges, y_edges);
% 创建热图
figure('Color', 'w');
imagesc(x_edges(1:end-1), y_edges(1:end-1), counts_2d');
set(gca, 'YDir', 'normal'); % 确保 y 轴方向正确
colorbar;
xlabel('Sample index');
ylabel('Tooth profile error');
title('Heat map of tooth profile error');
% 设置白色到蓝色渐变颜色映射
n = 256; % 颜色映射的分辨率
white_to_blue_map = [linspace(1, 0, n)', linspace(1, 0, n)', ones(n, 1)]; % 从白色到蓝色的渐变
colormap(white_to_blue_map);


%%
%y1y2热图

% 计算二维直方图
numBins = 100; % 设置直方图的边数
[counts, edges] = hist3([Y1_values_filtered, Y2_values_filtered], 'Nbins', [numBins numBins]);
% 平滑处理
counts = imgaussfilt(counts, 1); % 使用高斯滤波器进行平滑处理
% 创建图形窗口并设置背景颜色为白色
figure('Color', 'w');
% 绘制热度图
imagesc(edges{1}, edges{2}, counts');
set(gca, 'YDir', 'normal'); % 调整Y轴方向，使原点在左下角
% 自定义颜色映射：从白色到红色
customColormap = [linspace(1, 1, 256)', linspace(1, 0, 256)', linspace(1, 0, 256)'];
colormap(customColormap);
% 添加颜色条
colorbar;
% 设置图形背景颜色为白色
set(gca, 'Color', 'w');
% 添加标题和轴标签
title('Tooth profile error - Quality loss curve', 'FontSize', 14);
xlabel('Tooth profile error', 'FontSize', 12);
ylabel('Quality loss', 'FontSize', 12);
% 设置图形属性
set(gca, 'FontSize', 12, 'Box', 'on'); % 设置字体大小和边框
axis square; % 使坐标轴为方形
% 调整颜色条
c = colorbar;
c.Label.String = 'Density';
c.Label.FontSize = 12;
% 确保热度图背景为白色
set(gcf, 'InvertHardcopy', 'off');
 %%
% %绘制直方图
% % 绘制y1的直方图
% figure('Color', 'w'); % 创建一个新图形窗口
% histogram(Y1_values_filtered, 50); % 使用50个柱子来绘制直方图
% title('后验');
% xlabel('Y1 values');
% ylabel('Frequency');
% box off; % 关闭图形的边框线
% 
% % 绘制y1的直方图
% figure('Color', 'w'); % 创建一个新图形窗口
% histogram(Y1_prior_values, 50); % 使用50个柱子来绘制直方图
% title('先验');
% xlabel('Y1 values');
% ylabel('Frequency');
% box off; % 关闭图形的边框线
% 
% % 绘制y1的直方图
% figure('Color', 'w'); % 创建一个新图形窗口
% histogram(actual_Y1, 50); % 使用50个柱子来绘制直方图
% title('实际');
% xlabel('Y1 values');
% ylabel('Frequency');
% box off; % 关闭图形的边框线
%%
%绘制y1后验样本和实际样本比较
% 确保 Y1_values_filtered 和 actual_Y1 的长度一致，插值 actual_Y1
xi = linspace(1, length(actual_Y1), length(Y1_values_filtered));
actual_Y1_interpolated = interp1(1:length(actual_Y1), actual_Y1, xi);
% 绘制后验的直方图和核密度图
figure('Color', 'w'); % 创建一个新图形窗口
hold on; % 保持当前图形
% 绘制后验直方图
h1 = histogram(Y1_values_filtered, 20, 'Normalization', 'pdf');
h1.FaceColor = [0 0.4470 0.7410]; % 蓝色
h1.EdgeColor = 'none'; % 无外框线
h1.FaceAlpha = 0.5; % 设置透明度
% 绘制实际直方图
h2 = histogram(actual_Y1_interpolated, 20, 'Normalization', 'pdf');
h2.FaceColor = [0.8500 0.3250 0.0980]; % 橙色
h2.EdgeColor = 'none'; % 无外框线
h2.FaceAlpha = 0.5; % 设置透明度
% 计算并绘制后验的核密度图
[f1, xi1] = ksdensity(Y1_values_filtered); 
plot(xi1, f1, 'Color', [0 0.4470 0.7410], 'LineWidth', 2); % 核密度图用蓝色
% 计算并绘制实际的核密度图
[f2, xi2] = ksdensity(actual_Y1_interpolated); 
plot(xi2, f2, 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 2); % 核密度图用橙色
title('Comparison between the predicted and experimental values of the probability density function and kernel density estimation of tooth profile errors');
xlabel('Tooth profile error');
ylabel('Density');
legend('后验直方图', '实际直方图', '后验核密度图', '实际核密度图');
% 设置图例和轴的字体大小
set(gca, 'FontSize', 12);
legend('show', 'Location', 'best');
hold off; % 释放当前图形
%%
%计算概率密度比较值
% 计算对数似然比
mu_pred = mean(Y1_values_filtered);
sigma_pred = std(Y1_values_filtered);
log_likelihood_pred = -0.5 * sum((actual_Y1_interpolated - mu_pred).^2 / sigma_pred^2 + log(2 * pi * sigma_pred^2));
disp(['对数似然比: ', num2str(log_likelihood_pred)]);

% 计算 KL 散度
mu_actual = mean(actual_Y1_interpolated);
sigma_actual = std(actual_Y1_interpolated);

KL_divergence = log(sigma_actual/sigma_pred) + (sigma_pred^2 + (mu_pred - mu_actual)^2) / (2 * sigma_actual^2) - 0.5;
disp(['KL 散度: ', num2str(KL_divergence)]);

% 计算 CRPS
% 创建一个评估点范围
y_range = linspace(min(actual_Y1_interpolated)-5, max(actual_Y1_interpolated)+5, 1000);
F_pred = normcdf(y_range, mu_pred, sigma_pred);  % 预测分布的CDF
F_obs = normcdf(y_range, mean(actual_Y1_interpolated), std(actual_Y1_interpolated)); % 真实观测值的CDF

% 计算 CRPS
crps_value = trapz(y_range, (F_pred - F_obs).^2); % 使用数值积分计算 CRPS
disp(['CRPS: ', num2str(crps_value)]);
%%
%齿廓线预测
% 设置抽取的点数和数据集数
num_points = 37;
num_datasets = 4;

% 初始化存储抽取的点的数组
all_y1_values = zeros(num_points, num_datasets);

% 获取后验样本的最大值和最小值
posterior_samples = squeeze(Y1_values_filtered(:, :, end));
y1_max = max(Y1_values_filtered(:, 1));
% 重复抽取多组数据集并绘制散点图
for dataset = 1:num_datasets
    % 按照后验样本输出值 y1 的概率密度分布抽取 37 个点
    y1_values_sampled = zeros(num_points, 1);
    for i = 1:num_points
        % 使用蒙特卡洛方法从后验样本输出值 y1 的概率密度分布中抽取一个点
        y1_values_sampled(i) = randsample(posterior_samples(:, 1), 1);
    end
     
%         % 确保抽取的点包含 y1 的最大值和最小值
%     y1_values_sampled(end) = y1_max;
%     
%     all_y1_values(:, dataset) = y1_values_sampled;
%     
    % 记录散点坐标
    scatter_coordinates = [1:num_points; y1_values_sampled']';
    all_scatter_coordinates{dataset} = scatter_coordinates;
    
    % 绘制散点图
    figure('Color', 'w');
    scatter(1:num_points, y1_values_sampled, 'filled');
    xlabel('Data Point Index');
    ylabel('y1');
    title(['Scatter Plot of y1 for Dataset ', num2str(dataset)]);
    
    % 输出散点具体数据
    disp(['Scatter Coordinates for Dataset ', num2str(dataset), ':']);
    disp(scatter_coordinates);
    
end

%%
%拟合误差-质量损失函数曲线
% 对实际数据进行二次多项式拟合
p_actual_y1 = polyfit(actual_Y1, actual_Y2, 2);

% 生成用于绘图的实际数据拟合曲线
x_fit_actual_y1 = linspace(min(actual_Y1), max(actual_Y1), 100);
y_fit_actual_y1 = polyval(p_actual_y1, x_fit_actual_y1);

% 最小化目标函数：找到最佳的 a 使得总平方误差最小
objFun = @(a) sum((a * Y1_values_filtered.^2 - Y2_values_filtered).^2);

% 使用 fminsearch 找到最佳的 a
%   a_opt = fminsearch(objFun, 1);  % 初始猜测为 1
  a_opt = 6.907735433500013e+08;
% 使用得到的系数 a_opt 生成拟合曲线
x_fit = linspace(min(Y1_values_filtered), max(Y1_values_filtered), 100);
y_fit = a_opt * x_fit.^2;

% 绘制实际数据散点及拟合曲线
figure('Color', 'w');
% plot(actual_Y1_filtered, actual_Y2_filtered, 'ro');  % 实际数据的散点图
hold on;
plot(x_fit_actual_y1, y_fit_actual_y1, 'r-');  % 实际数据的拟合曲线

% 绘制后验样本数据散点及拟合曲线
% plot(Y1_values_filtered, Y2_values_filtered, 'bo');  % 后验样本的散点图
plot(x_fit, y_fit, 'b-');  % 后验样本的拟合曲线

% 设置图例和标题
legend('Actual Data', 'Actual Quadratic Fit');
title('Tooth profile error - Quality loss curve');
xlabel('Tooth profile error');
ylabel('Quality loss');
box off; % 关闭图形的边框线
hold off;
%%
% %拟合误差-质量损失函数曲线（部分）
% % 对实际数据进行二次多项式拟合
% p_actual_y1 = polyfit(actual_Y1, actual_Y2, 2);
% 
% % 生成用于绘图的实际数据拟合曲线
% x_fit_actual_y1 = linspace(min(actual_Y1), max(actual_Y1), 100);
% y_fit_actual_y1 = polyval(p_actual_y1, x_fit_actual_y1);
% 
% % 最小化目标函数：找到最佳的 a 使得总平方误差最小
% objFun = @(a) sum((a * Y1_values_filtered.^2 - Y2_values_filtered).^2);
% 
% % 使用 fminsearch 找到最佳的 a
% % a_opt = fminsearch(objFun, 1);  % 初始猜测为 1
% a_opt = 6.907735433500013e+08;
% 
% % 获取actual_Y1的最大值和最小值
% min_actual_Y1 = min(actual_Y1);
% max_actual_Y1 = max(actual_Y1);
% 
% % 只在 actual_Y1 的最大值和最小值之间生成拟合曲线
% x_fit = linspace(min_actual_Y1, max_actual_Y1, 100);
% y_fit = a_opt * x_fit.^2;
% 
% % 绘制实际数据散点及拟合曲线
% figure('Color', 'w');
% % plot(actual_Y1_filtered, actual_Y2_filtered, 'ro');  % 实际数据的散点图
% hold on;
% plot(x_fit_actual_y1, y_fit_actual_y1, 'r-');  % 实际数据的拟合曲线
% 
% % 绘制后验样本数据散点及拟合曲线
% % plot(Y1_values_filtered, Y2_values_filtered, 'bo');  % 后验样本的散点图
% plot(x_fit, y_fit, 'b-');  % 后验样本的拟合曲线
% 
% % 设置图例和标题
% legend('Actual Data', 'Actual Quadratic Fit');
% title('Tooth profile error - Quality loss curve');
% xlabel('Tooth profile error');
% ylabel('Quality loss');
% box off; % 关闭图形的边框线
% hold off;

% 结束计时并输出经过的时间
elapsed_time = toc;
disp(['代码执行时间: ', num2str(elapsed_time), ' 秒']);
