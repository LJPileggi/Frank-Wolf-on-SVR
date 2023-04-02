function ModelSel(K, train_alg, C_list, eps_list, sigma_list, varargin)
    hyperparams = cartprod(C_list, eps_list, sigma_list);
    lag = 5;
    dataset = readtable('energydata.csv');
    dataset = table2array(dataset);
    train_set = dataset(1:fix(size(dataset, 1)*0.7),:);
    train_set = train_set.';
    X = [];
    for i = lag:size(train_set, 1)-1
        X = [X ; train_set(i-lag+1:i)];
    end
    Y = train_set(lag+1:end);
    for i = 1:K
        KFold(hyperparams(i,:), train_alg, X, Y, K, varargin);
    end
end