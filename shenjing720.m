clc
clear all
close all
% ��ʼ��ʱ
tic;
% ��ȡ����
dataO1 = xlsread('C:\Users\Administrator.DESKTOP-JC66FHP\Desktop\�½��ļ���\�ӹ����Ԥ��С����\С���ĳ���\1019 - ����-1\ʵ������2.xlsx');

% ���������������������������
num_input_features = 18;
num_output_features = 2;

% ѡ��ǰ18����Ϊ������������������Ϊ�������
x_feature_label = dataO1(:, 1:num_input_features);
y_feature_label = dataO1(:, num_input_features + 1:num_input_features + num_output_features);

% ����ѵ������Ԥ�⼯
train_data = dataO1(1:480, :);
test_data = dataO1(481:500, :);

% ��ȡѵ�����Ͳ��Լ���������������
train_x_feature_label = train_data(:, 1:num_input_features);
train_y_feature_label = train_data(:, num_input_features + 1:num_input_features + num_output_features);

test_x_feature_label = test_data(:, 1:num_input_features);
test_y_feature_label = test_data(:, num_input_features + 1:num_input_features + num_output_features);

% ��׼��ѵ�����Ͳ��Լ���������������
x_mu = mean(train_x_feature_label);
x_sig = std(train_x_feature_label);
train_x_feature_label_norm = (train_x_feature_label - x_mu) ./ x_sig;
test_x_feature_label_norm = (test_x_feature_label - x_mu) ./ x_sig;

y_mu = mean(train_y_feature_label);
y_sig = std(train_y_feature_label);
train_y_feature_label_norm = (train_y_feature_label - y_mu) ./ y_sig;

% ����������ģ��
% �������ز㣬ÿ���10����Ԫ
hidden_layer_sizes = [10, 10];  % ���ز��С������
Mdl = newff(train_x_feature_label_norm', train_y_feature_label_norm', hidden_layer_sizes, {'logsig', 'logsig', 'purelin'});

% ѵ��������ģ��
[Mdl, ~] = train(Mdl, train_x_feature_label_norm', train_y_feature_label_norm');

% ʹ��ѵ���õ�ģ�ͽ���Ԥ��
y_test_predict_norm = sim(Mdl, test_x_feature_label_norm')';
y_test_predict = y_test_predict_norm .* y_sig + y_mu;

% % ���� y1 ��Ԥ��ֵ��ʵ��ֵ�Ƚ�ͼ
% subplot(2, 1, 1);  % ��Ϊ����һ�е�ͼ�εĵ�һ��
% plot(test_y_feature_label(:,1), 'o-', 'LineWidth', 2, 'MarkerSize', 6);  % ʵ��ֵ
% hold on;
% plot(y_test_predict(:,1), 'x--', 'LineWidth', 2, 'MarkerSize', 6);  % Ԥ��ֵ
% title('Comparison of Actual and Predicted Y1');
% xlabel('Sample Index');
% ylabel('Y1 Value');
% legend('Actual Y1', 'Predicted Y1');
% grid on;
% hold off;
% 
% % ���� y2 ��Ԥ��ֵ��ʵ��ֵ�Ƚ�ͼ
% subplot(2, 1, 2);  % ��Ϊ����һ�е�ͼ�εĵڶ���
% plot(test_y_feature_label(:,2), 'o-', 'LineWidth', 2, 'MarkerSize', 6);  % ʵ��ֵ
% hold on;
% plot(y_test_predict(:,2), 'x--', 'LineWidth', 2, 'MarkerSize', 6);  % Ԥ��ֵ
% title('Comparison of Actual and Predicted Y2');
% xlabel('Sample Index');
% ylabel('Y2 Value');
% legend('Actual Y2', 'Predicted Y2');
% grid on;
% hold off;

% % ��ʾͼ��
% figure(gcf);

%%
rng(123);  % �趨��������Ի�ÿ��ظ��Ľ��

% ��������(ǰ18��Ϊ����)
nParameters = 18;

% % �������½�
% param_lb =[-0.001, -0.001, -0.001, -0.001, -0.001,  -0.001, -0.002,-0.010, -0.005, 0, 0, 0, 0, 0, 0, 0, 0, 0]; % �����½�
% param_ub =[0.001, 0.001, 0.001, 0.001,0.001,0.001, 0.002, 0.010, 0.006, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1, 1, 1]; % �����Ͻ�
% �������½�
param_lb =[-0.001, -0.001, -0.001, -0.001, -0.001,  -0.001, 0,-0.0011, -0.0044,0, 0, 0, 0, 0, 0, 0, 0, 0]; % �����½�
param_ub =[0.001, 0.001, 0.001, 0.001,0.001,0.001, 0.0012, 0.0026, -0.002, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1, 1, 1]; % �����Ͻ�

% �趨MCMC����
nIterations1 = 500; % ���������ĵ�������
nBurnIn1 = 100; % ����������ȼ����
nChains1 = 500; % ��������������

nIterations2 =500; % ���������ĵ�������
nBurnIn2 = 100; % ����������ȼ����
nChains2 = 500; % ��������������

% ��ʼ��MCMC������ʼֵ�������½緶Χ�������ʼ����
initial_values = bsxfun(@plus, param_lb, bsxfun(@times, param_ub - param_lb, rand(nChains1 + nChains2, nParameters)));
% �洢MCMC��������
chain_samples1 = NaN(nIterations1, nChains1, nParameters);
chain_samples2 = NaN(nIterations2, nChains2, nParameters);
% �洢����ֲ�����
posterior_samples = NaN(nChains2, nParameters);

% �洢����������Ŀ�꺯��ֵ
model_outputs_prior = NaN(nChains1, 2, nIterations1);  % 2 ��ʾĿ�꺯�����������

% �����ֵ�ͱ�׼��
param_mean = (param_lb + param_ub) / 2; 
param_std = (param_ub - param_lb) / (2 * sqrt(3)); 

% �洢��ֵ�ͱ�׼��
for i = 1:length(param_mean)
    eval(['mu_param', num2str(i), ' = param_mean(i);']);
    eval(['sigma_param', num2str(i), ' = param_std(i);']);
end

% �洢��ֵ�ͱ�׼��
mu_prior = [mu_param1, mu_param2, mu_param3, mu_param4, mu_param5, mu_param6, mu_param7, mu_param8, mu_param9, mu_param10, mu_param11, mu_param12, mu_param13, mu_param14, mu_param15, mu_param16, mu_param17, mu_param18];
sigma_prior = [sigma_param1, sigma_param2, sigma_param3, sigma_param4, sigma_param5, sigma_param6, sigma_param7, sigma_param8, sigma_param9, sigma_param10, sigma_param11, sigma_param12, sigma_param13, sigma_param14, sigma_param15, sigma_param16, sigma_param17, sigma_param18];

% ������������
for chain1 = 1:nChains1
    current_params1 = initial_values(chain1, :);
    accepted_samples1 = zeros(nIterations1- nBurnIn1, nParameters);
    count1 = 1;
    for iteration1 = 1:nIterations1
        proposed_params1 = current_params1;
        % ��̬�ֲ��������²���
        proposed_params1(1:9) = normrnd(mu_prior(1:9), sigma_prior(1:9)); 
        proposed_params1(10:18) = normrnd(mu_prior(10:18), sigma_prior(10:18)); 
        
        % ģ��Ԥ��
        y_current_norm1 = sim(Mdl, proposed_params1')';
        Y_current1 = y_current_norm1 .* y_sig + y_mu; % ����һ��

        % Ӧ��Լ������
        if Y_current1(1) == 0
            Y_current1(2) = 0;
        else
            Y_current1(2) = max(0, Y_current1(2));
        end

        % �洢ģ�����
        model_outputs_prior(iteration1, 1, chain1) = Y_current1(:, 1);
        model_outputs_prior(iteration1, 2, chain1) = Y_current1(:, 2);

        % ���²���
        current_params1 = proposed_params1;
        if iteration1 > nBurnIn1
            accepted_samples1(count1, :) = proposed_params1;
            count1 = count1 + 1;
        end
    end
    chain_samples1(1:nIterations1 - nBurnIn1, chain1, :) = accepted_samples1;
end

% % ������������
% for chain1 = 1:nChains1
%     current_params1 = initial_values(chain1, :);
%     accepted_samples1 = zeros(nIterations1 - nBurnIn1, nParameters);
%     count1 = 1;
%     for iteration1 = 1:nIterations1
%         proposed_params1 = current_params1;
%         % ��̬�ֲ��������²���
%         proposed_params1(1:9) = normrnd(mu_prior(1:9), sigma_prior(1:9)); 
%         proposed_params1(10:18) = normrnd(mu_prior(10:18), sigma_prior(10:18)); 
%         
%         % ģ��Ԥ��
%         y_current_norm1 = sim(Mdl, proposed_params1')';
%         Y_current1 = y_current_norm1 .* y_sig + y_mu; % ����һ��
% 
%         % Ӧ��Լ������
%         if Y_current1(1) == 0
%             Y_current1(2) = 0;
%         else
%             Y_current1(2) = max(0, Y_current1(2));
%         end
% 
%         % �洢ģ�����
%         model_outputs_prior(iteration1, 1, chain1) = Y_current1(:, 1);
%         model_outputs_prior(iteration1, 2, chain1) = Y_current1(:, 2);
% 
%         % ���²���
%         % ֻ��y2����0ʱ�Ÿ��ºʹ洢����
%         if Y_current1(2) > 0
%             current_params1 = proposed_params1;
%             if iteration1 > nBurnIn1
%                 accepted_samples1(count1, :) = proposed_params1;
%                 count1 = count1 + 1;
%             end
%         end
%     end
%     % �洢ÿ�����Ĳ������������Է�ȼ���ڵĵ���
%     chain_samples1(1:count1-1, chain1, :) = accepted_samples1(1:count1-1, :);
% end

% 
% ����۲�����
% actual_data = xlsread('14.xls');
actual_data = xlsread('13.xls');
actual_Y1 = actual_data(1:148, 1);
actual_Y2 = actual_data(1:148, 2);

% ����actual_data��y1��y2�ľ�ֵ
mean_actual_Y1 = mean(actual_data(1:148, 1));
mean_actual_Y2 = mean(actual_data(1:148, 2));

% ������������
model_outputs_posterior = NaN(nIterations2, 2, nChains2); % �洢����������Ŀ�꺯

% �����Ѿ��� actual_Y1 ������KDE������ȡ����������ܶ�ֵ
[f_Y1, xi_Y1] = ksdensity(actual_Y1);

% ���� param_lb �� param_ub �ֱ��ǲ������½���Ͻ�
param_range = param_ub - param_lb;

% ��ʼ����ʹ�ò�����Χ��һ��С������Ϊ����
step_size = 0.1 * param_range;

for chain2 = 1:nChains2
    current_params2 = initial_values(nChains1 + chain2, :);
    accepted_samples2 = zeros(nIterations2 - nBurnIn2, nParameters);
     count2 = 1;
    for iteration = 1:nIterations2
        % ʹ��������ģ�ͽ��е�ǰ������Ԥ��
        y_current_norm = sim(Mdl, current_params2')';
        Y_current = y_current_norm .* y_sig + y_mu; % ����һ��
        % Ӧ��Լ������
        if Y_current(1) == 0
            Y_current(2) = 0;
        else
            Y_current(2) = max(0, Y_current(2));
        end
        % ���㵱ǰ״̬����y1����Ȼֵ
        likelihood_current_Y1 = interp1(xi_Y1, f_Y1, Y_current(:, 1), 'linear', 0);
        likelihood_current = likelihood_current_Y1;
             
        % �����µ����������ͨ���ڵ�ǰ�����Ļ����ϼ�������Ŷ�
        proposed_params2 = current_params2 + normrnd(0, step_size, [1, nParameters]);
           
        % ������Ĳ�������ģ��Ԥ��
        y_proposed_norm = sim(Mdl, proposed_params2')';
        Y_proposed = y_proposed_norm .* y_sig + y_mu; % ����һ��
        % Ӧ��Լ������
        if Y_proposed(1) == 0
            Y_proposed(2) = 0;
        else
            Y_proposed(2) = max(0, Y_proposed(2));
        end
        
        % ��������״̬����y1����Ȼֵ
        likelihood_proposed_Y1 = interp1(xi_Y1, f_Y1, Y_proposed(:, 1), 'linear', 0);
        likelihood_proposed = likelihood_proposed_Y1;
       
        % ��y1��y2��ģ������洢��model_outputs_posterior��
        model_outputs_posterior(iteration, 1, chain2) = Y_proposed(:, 1);
        model_outputs_posterior(iteration, 2, chain2) = Y_proposed(:, 2);
        
        % ����������ʣ�ע������ʹ��proposed_params2
        prior_current = prod(normpdf(current_params1, mu_prior, sigma_prior));
        prior_proposed = prod(normpdf(proposed_params1, mu_prior, sigma_prior));
        
        % ���������
        acceptance_ratio = (likelihood_proposed * prior_proposed) / (likelihood_current * prior_current);

        if rand() < acceptance_ratio
            current_params2 = proposed_params2;
        end
        
        if iteration > nBurnIn2
            accepted_samples2(count2, :) = current_params2;
            count2 = count2 + 1;
        end
    end

% �洢ÿ�����Ĳ������������Է�ȼ���ڵĵ���
chain_samples2(1:nIterations2 - nBurnIn2, chain2, :) = accepted_samples2;

end

% for chain2 = 1:nChains2
%     current_params2 = initial_values(nChains1 + chain2, :);
%     accepted_samples2 = zeros(nIterations2 - nBurnIn2, nParameters);
%     count2 = 1;
%     for iteration2 = 1:nIterations2
%         % ʹ��������ģ�ͽ��е�ǰ������Ԥ��
%         y_current_norm = sim(Mdl, current_params2')';
%         Y_current = y_current_norm .* y_sig + y_mu; % ����һ��
%         % Ӧ��Լ������
%         if Y_current(1) == 0
%             Y_current(2) = 0;
%         else
%             Y_current(2) = max(0, Y_current(2));
%         end
%         % ���㵱ǰ״̬����y1����Ȼֵ
%         likelihood_current_Y1 = interp1(xi_Y1, f_Y1, Y_current(:, 1), 'linear', 0);
%         likelihood_current = likelihood_current_Y1;
%              
%         % �����µ����������ͨ���ڵ�ǰ�����Ļ����ϼ�������Ŷ�
%         proposed_params2 = current_params2 + normrnd(0, step_size, [1, nParameters]);
%            
%         % ������Ĳ�������ģ��Ԥ��
%         y_proposed_norm = sim(Mdl, proposed_params2')';
%         Y_proposed = y_proposed_norm .* y_sig + y_mu; % ����һ��
%         % Ӧ��Լ������
%         if Y_proposed(1) == 0
%             Y_proposed(2) = 0;
%         else
%             Y_proposed(2) = max(0, Y_proposed(2));
%         end
%         
%         % ��������״̬����y1����Ȼֵ
%         likelihood_proposed_Y1 = interp1(xi_Y1, f_Y1, Y_proposed(:, 1), 'linear', 0);
%         likelihood_proposed = likelihood_proposed_Y1;
%        
%         % ��y1��y2��ģ������洢��model_outputs_posterior��
%         if Y_proposed(2) > 0
%             model_outputs_posterior(iteration2, 1, chain2) = Y_proposed(:, 1);
%             model_outputs_posterior(iteration2, 2, chain2) = Y_proposed(:, 2);
%         end
%         
%         % ����������ʣ�ע������ʹ��proposed_params2
%         prior_current = prod(normpdf(current_params1, mu_prior, sigma_prior));
%         prior_proposed = prod(normpdf(proposed_params1, mu_prior, sigma_prior));
%         
%         % ���������
%         acceptance_ratio = (likelihood_proposed * prior_proposed) / (likelihood_current * prior_current);
% 
%         if rand() < acceptance_ratio
%             current_params2 = proposed_params2;
%         end
%         
%         if iteration2 > nBurnIn2
%             % ֻ��y2����0ʱ�ű�������
%             if Y_proposed(2) > 0
%                 accepted_samples2(count2, :) = current_params2;
%                 count2 = count2 + 1;
%             end
%         end
%     end
% 
%     % �洢ÿ�����Ĳ������������Է�ȼ���ڵĵ���
%     chain_samples2(1:nIterations2 - nBurnIn2, chain2, :) = accepted_samples2;
% 
% end

% ����MCMC���
mean_params_prior = squeeze(mean(chain_samples1, 1));
std_params_prior = squeeze(std(chain_samples1, 1));

mean_params_posterior = squeeze(mean(chain_samples2, 1));
std_params_posterior = squeeze(std(chain_samples2, 1));

% ��ȡy1��y2�ĺ�������ֵ
Y1_values = reshape(model_outputs_posterior(:, 1, :), [], 1); % ����������y1ֵ��ȡ��һ��������
Y2_values = reshape(model_outputs_posterior(:, 2, :), [], 1); % ����������y2ֵ��ȡ��һ��������

% ɸѡ��y2 > 0������
valid_indices = Y2_values > 0;
Y1_values_filtered = Y1_values(valid_indices);
Y2_values_filtered = Y2_values(valid_indices);

%%
%y1��ͼ
% ��������
x_values = (1:length(Y1_values_filtered))';
% ���������
num_bins_x = 20; % ���������ķ�����
num_bins_y = 20; % ����ֵ�ķ�����
% ��������߽�
x_edges = linspace(min(x_values), max(x_values), num_bins_x + 1);
y_edges = linspace(min(Y1_values_filtered), max(Y1_values_filtered), num_bins_y + 1);
% �����άֱ��ͼ
[counts_2d, ~, ~] = histcounts2(x_values, Y1_values_filtered, x_edges, y_edges);
% ������ͼ
figure('Color', 'w');
imagesc(x_edges(1:end-1), y_edges(1:end-1), counts_2d');
set(gca, 'YDir', 'normal'); % ȷ�� y �᷽����ȷ
colorbar;
xlabel('Sample index');
ylabel('Tooth profile error');
title('Heat map of tooth profile error');
% ���ð�ɫ����ɫ������ɫӳ��
n = 256; % ��ɫӳ��ķֱ���
white_to_blue_map = [linspace(1, 0, n)', linspace(1, 0, n)', ones(n, 1)]; % �Ӱ�ɫ����ɫ�Ľ���
colormap(white_to_blue_map);


%%
%y1y2��ͼ

% �����άֱ��ͼ
numBins = 100; % ����ֱ��ͼ�ı���
[counts, edges] = hist3([Y1_values_filtered, Y2_values_filtered], 'Nbins', [numBins numBins]);
% ƽ������
counts = imgaussfilt(counts, 1); % ʹ�ø�˹�˲�������ƽ������
% ����ͼ�δ��ڲ����ñ�����ɫΪ��ɫ
figure('Color', 'w');
% �����ȶ�ͼ
imagesc(edges{1}, edges{2}, counts');
set(gca, 'YDir', 'normal'); % ����Y�᷽��ʹԭ�������½�
% �Զ�����ɫӳ�䣺�Ӱ�ɫ����ɫ
customColormap = [linspace(1, 1, 256)', linspace(1, 0, 256)', linspace(1, 0, 256)'];
colormap(customColormap);
% �����ɫ��
colorbar;
% ����ͼ�α�����ɫΪ��ɫ
set(gca, 'Color', 'w');
% ��ӱ�������ǩ
title('Tooth profile error - Quality loss curve', 'FontSize', 14);
xlabel('Tooth profile error', 'FontSize', 12);
ylabel('Quality loss', 'FontSize', 12);
% ����ͼ������
set(gca, 'FontSize', 12, 'Box', 'on'); % ���������С�ͱ߿�
axis square; % ʹ������Ϊ����
% ������ɫ��
c = colorbar;
c.Label.String = 'Density';
c.Label.FontSize = 12;
% ȷ���ȶ�ͼ����Ϊ��ɫ
set(gcf, 'InvertHardcopy', 'off');
 %%
% %����ֱ��ͼ
% % ����y1��ֱ��ͼ
% figure('Color', 'w'); % ����һ����ͼ�δ���
% histogram(Y1_values_filtered, 50); % ʹ��50������������ֱ��ͼ
% title('����');
% xlabel('Y1 values');
% ylabel('Frequency');
% box off; % �ر�ͼ�εı߿���
% 
% % ����y1��ֱ��ͼ
% figure('Color', 'w'); % ����һ����ͼ�δ���
% histogram(Y1_prior_values, 50); % ʹ��50������������ֱ��ͼ
% title('����');
% xlabel('Y1 values');
% ylabel('Frequency');
% box off; % �ر�ͼ�εı߿���
% 
% % ����y1��ֱ��ͼ
% figure('Color', 'w'); % ����һ����ͼ�δ���
% histogram(actual_Y1, 50); % ʹ��50������������ֱ��ͼ
% title('ʵ��');
% xlabel('Y1 values');
% ylabel('Frequency');
% box off; % �ر�ͼ�εı߿���
%%
%����y1����������ʵ�������Ƚ�
% ȷ�� Y1_values_filtered �� actual_Y1 �ĳ���һ�£���ֵ actual_Y1
xi = linspace(1, length(actual_Y1), length(Y1_values_filtered));
actual_Y1_interpolated = interp1(1:length(actual_Y1), actual_Y1, xi);
% ���ƺ����ֱ��ͼ�ͺ��ܶ�ͼ
figure('Color', 'w'); % ����һ����ͼ�δ���
hold on; % ���ֵ�ǰͼ��
% ���ƺ���ֱ��ͼ
h1 = histogram(Y1_values_filtered, 20, 'Normalization', 'pdf');
h1.FaceColor = [0 0.4470 0.7410]; % ��ɫ
h1.EdgeColor = 'none'; % �������
h1.FaceAlpha = 0.5; % ����͸����
% ����ʵ��ֱ��ͼ
h2 = histogram(actual_Y1_interpolated, 20, 'Normalization', 'pdf');
h2.FaceColor = [0.8500 0.3250 0.0980]; % ��ɫ
h2.EdgeColor = 'none'; % �������
h2.FaceAlpha = 0.5; % ����͸����
% ���㲢���ƺ���ĺ��ܶ�ͼ
[f1, xi1] = ksdensity(Y1_values_filtered); 
plot(xi1, f1, 'Color', [0 0.4470 0.7410], 'LineWidth', 2); % ���ܶ�ͼ����ɫ
% ���㲢����ʵ�ʵĺ��ܶ�ͼ
[f2, xi2] = ksdensity(actual_Y1_interpolated); 
plot(xi2, f2, 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 2); % ���ܶ�ͼ�ó�ɫ
title('Comparison between the predicted and experimental values of the probability density function and kernel density estimation of tooth profile errors');
xlabel('Tooth profile error');
ylabel('Density');
legend('����ֱ��ͼ', 'ʵ��ֱ��ͼ', '������ܶ�ͼ', 'ʵ�ʺ��ܶ�ͼ');
% ����ͼ������������С
set(gca, 'FontSize', 12);
legend('show', 'Location', 'best');
hold off; % �ͷŵ�ǰͼ��
%%
%��������ܶȱȽ�ֵ
% ���������Ȼ��
mu_pred = mean(Y1_values_filtered);
sigma_pred = std(Y1_values_filtered);
log_likelihood_pred = -0.5 * sum((actual_Y1_interpolated - mu_pred).^2 / sigma_pred^2 + log(2 * pi * sigma_pred^2));
disp(['������Ȼ��: ', num2str(log_likelihood_pred)]);

% ���� KL ɢ��
mu_actual = mean(actual_Y1_interpolated);
sigma_actual = std(actual_Y1_interpolated);

KL_divergence = log(sigma_actual/sigma_pred) + (sigma_pred^2 + (mu_pred - mu_actual)^2) / (2 * sigma_actual^2) - 0.5;
disp(['KL ɢ��: ', num2str(KL_divergence)]);

% ���� CRPS
% ����һ�������㷶Χ
y_range = linspace(min(actual_Y1_interpolated)-5, max(actual_Y1_interpolated)+5, 1000);
F_pred = normcdf(y_range, mu_pred, sigma_pred);  % Ԥ��ֲ���CDF
F_obs = normcdf(y_range, mean(actual_Y1_interpolated), std(actual_Y1_interpolated)); % ��ʵ�۲�ֵ��CDF

% ���� CRPS
crps_value = trapz(y_range, (F_pred - F_obs).^2); % ʹ����ֵ���ּ��� CRPS
disp(['CRPS: ', num2str(crps_value)]);
%%
%������Ԥ��
% ���ó�ȡ�ĵ��������ݼ���
num_points = 37;
num_datasets = 4;

% ��ʼ���洢��ȡ�ĵ������
all_y1_values = zeros(num_points, num_datasets);

% ��ȡ�������������ֵ����Сֵ
posterior_samples = squeeze(Y1_values_filtered(:, :, end));
y1_max = max(Y1_values_filtered(:, 1));
% �ظ���ȡ�������ݼ�������ɢ��ͼ
for dataset = 1:num_datasets
    % ���պ����������ֵ y1 �ĸ����ܶȷֲ���ȡ 37 ����
    y1_values_sampled = zeros(num_points, 1);
    for i = 1:num_points
        % ʹ�����ؿ��巽���Ӻ����������ֵ y1 �ĸ����ܶȷֲ��г�ȡһ����
        y1_values_sampled(i) = randsample(posterior_samples(:, 1), 1);
    end
     
%         % ȷ����ȡ�ĵ���� y1 �����ֵ����Сֵ
%     y1_values_sampled(end) = y1_max;
%     
%     all_y1_values(:, dataset) = y1_values_sampled;
%     
    % ��¼ɢ������
    scatter_coordinates = [1:num_points; y1_values_sampled']';
    all_scatter_coordinates{dataset} = scatter_coordinates;
    
    % ����ɢ��ͼ
    figure('Color', 'w');
    scatter(1:num_points, y1_values_sampled, 'filled');
    xlabel('Data Point Index');
    ylabel('y1');
    title(['Scatter Plot of y1 for Dataset ', num2str(dataset)]);
    
    % ���ɢ���������
    disp(['Scatter Coordinates for Dataset ', num2str(dataset), ':']);
    disp(scatter_coordinates);
    
end

%%
%������-������ʧ��������
% ��ʵ�����ݽ��ж��ζ���ʽ���
p_actual_y1 = polyfit(actual_Y1, actual_Y2, 2);

% �������ڻ�ͼ��ʵ�������������
x_fit_actual_y1 = linspace(min(actual_Y1), max(actual_Y1), 100);
y_fit_actual_y1 = polyval(p_actual_y1, x_fit_actual_y1);

% ��С��Ŀ�꺯�����ҵ���ѵ� a ʹ����ƽ�������С
objFun = @(a) sum((a * Y1_values_filtered.^2 - Y2_values_filtered).^2);

% ʹ�� fminsearch �ҵ���ѵ� a
%   a_opt = fminsearch(objFun, 1);  % ��ʼ�²�Ϊ 1
  a_opt = 6.907735433500013e+08;
% ʹ�õõ���ϵ�� a_opt �����������
x_fit = linspace(min(Y1_values_filtered), max(Y1_values_filtered), 100);
y_fit = a_opt * x_fit.^2;

% ����ʵ������ɢ�㼰�������
figure('Color', 'w');
% plot(actual_Y1_filtered, actual_Y2_filtered, 'ro');  % ʵ�����ݵ�ɢ��ͼ
hold on;
plot(x_fit_actual_y1, y_fit_actual_y1, 'r-');  % ʵ�����ݵ��������

% ���ƺ�����������ɢ�㼰�������
% plot(Y1_values_filtered, Y2_values_filtered, 'bo');  % ����������ɢ��ͼ
plot(x_fit, y_fit, 'b-');  % �����������������

% ����ͼ���ͱ���
legend('Actual Data', 'Actual Quadratic Fit');
title('Tooth profile error - Quality loss curve');
xlabel('Tooth profile error');
ylabel('Quality loss');
box off; % �ر�ͼ�εı߿���
hold off;
%%
% %������-������ʧ�������ߣ����֣�
% % ��ʵ�����ݽ��ж��ζ���ʽ���
% p_actual_y1 = polyfit(actual_Y1, actual_Y2, 2);
% 
% % �������ڻ�ͼ��ʵ�������������
% x_fit_actual_y1 = linspace(min(actual_Y1), max(actual_Y1), 100);
% y_fit_actual_y1 = polyval(p_actual_y1, x_fit_actual_y1);
% 
% % ��С��Ŀ�꺯�����ҵ���ѵ� a ʹ����ƽ�������С
% objFun = @(a) sum((a * Y1_values_filtered.^2 - Y2_values_filtered).^2);
% 
% % ʹ�� fminsearch �ҵ���ѵ� a
% % a_opt = fminsearch(objFun, 1);  % ��ʼ�²�Ϊ 1
% a_opt = 6.907735433500013e+08;
% 
% % ��ȡactual_Y1�����ֵ����Сֵ
% min_actual_Y1 = min(actual_Y1);
% max_actual_Y1 = max(actual_Y1);
% 
% % ֻ�� actual_Y1 �����ֵ����Сֵ֮�������������
% x_fit = linspace(min_actual_Y1, max_actual_Y1, 100);
% y_fit = a_opt * x_fit.^2;
% 
% % ����ʵ������ɢ�㼰�������
% figure('Color', 'w');
% % plot(actual_Y1_filtered, actual_Y2_filtered, 'ro');  % ʵ�����ݵ�ɢ��ͼ
% hold on;
% plot(x_fit_actual_y1, y_fit_actual_y1, 'r-');  % ʵ�����ݵ��������
% 
% % ���ƺ�����������ɢ�㼰�������
% % plot(Y1_values_filtered, Y2_values_filtered, 'bo');  % ����������ɢ��ͼ
% plot(x_fit, y_fit, 'b-');  % �����������������
% 
% % ����ͼ���ͱ���
% legend('Actual Data', 'Actual Quadratic Fit');
% title('Tooth profile error - Quality loss curve');
% xlabel('Tooth profile error');
% ylabel('Quality loss');
% box off; % �ر�ͼ�εı߿���
% hold off;

% ������ʱ�����������ʱ��
elapsed_time = toc;
disp(['����ִ��ʱ��: ', num2str(elapsed_time), ' ��']);
