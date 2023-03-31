function [X_train, X_val, Y_train, Y_val] = KFoldSplitting(X, Y, K)
    X_train = {};
    X_val = {};
    Y_train = {};
    Y_val = {};
    for i = 1:K
        if i == 1
            X_train = {X_train , X(i*fix(size(X, 1)/K)+1:end,:)};
            X_val = {X_val , X(1:i*fix(size(X, 1)/K),:)};
            Y_train = {Y_train , Y(i*fix(size(X, 1)/K)+1:end)};
            Y_val = {Y_val , Y(1:i*fix(size(X, 1)/K))};
        elseif i == K
            X_train = {X_train , X(1:(i-1)*fix(size(X, 1)/K),:)};
            X_val = {X_val , X((i-1)*fix(size(X, 1)/K)+1:end,:)};
            Y_train = {Y_train , Y(1:(i-1)*fix(size(X, 1)/K))};
            Y_val = {Y_val , Y((i-1)*fix(size(X, 1)/K)+1:end)};
        else
            X_train = {X_train , [X(1:(i-1)*fix(size(X, 1)/K),:) ; X(i*fix(size(X, 1)/K)+1:end,:)]};
            X_val = {X_val , X((i-1)*fix(size(X, 1)/K)+1:i*fix(size(X, 1)/K),:)};
            Y_train = {Y_train , [Y_val(1:(i-1)*fix(size(X, 1)/K)) ; Y(i*fix(size(X, 1)/K)+1:end)]};
            Y_val = {Y_val , Y((i-1)*fix(size(X, 1)/K)+1:i*fix(size(X, 1)/K))};
        end
    end
end

function KFold(model_params, train_alg, X, Y, K, varargin)
    if isempty(varargin)
        fw_eps = 1e-3;
        fw_tau = 0.99;
        fw_N_max = 1000;
    elseif length(varargin) == 1
        fw_eps = varargin(1);
        fw_tau = 0.99;
        fw_N_max = 1000;
    elseif length(varargin) == 2
        fw_eps = varargin(1);
        fw_tau = varargin(2);
        fw_N_max = 1000;
    elseif length(varargin) == 3
        fw_eps = varargin(1);
        fw_tau = varargin(2);
        fw_N_max = varargin(3);
    end
    fw_params = {fw_eps , fw_tau , fw_N_Max};
    [X_train, X_val, Y_train, Y_val] = KFoldSplitting(X, Y, K);
    mse = [];
    for i = 1:K
        model = SVR(X_train{i}, Y_train{i}, model_params{:});
        alpha_opt = FrankWolfe(model.objfunc, model.objgrad, model.A, model.b, model.A_eq, model.b_eq, model.alpha, fw_params);
        model.alpha = alpha_opt;
        Y_predict = model.predict(X_val);
        mse = [mse ; (Y_predict - Y_val).^2/size(Y_val, 1)];
    end
    mean_mse = mean(mse);
    stdev_mse = std(mse);
    fprintf("Training completed, hyperparameters: %f\n", model_params);
    fprintf("mse: %f +- %f\n", mean_mse, stdev_mse);
end