function KFold(model_params, train_alg, X, Y, K, varargin)
    if isempty(varargin)
        eps = 1e-3;
        m1 = 1e-2;
        tau = 0.99;
        N_max = 1000;
        show = 10;
    elseif length(varargin) == 1
        eps = varargin(1);
        m1 = 1e-2;
        tau = 0.99;
        N_max = 1000;
        show = 10;
    elseif length(varargin) == 2
        eps = varargin(1);
        m1 = varargin(2);
        tau = 0.99;
        N_max = 1000;
        show = 10;
    elseif length(varargin) == 3
        eps = varargin(1);
        m1 = varargin(2);
        tau = varargin(3);
        N_max = 1000;
        show = 10;
    elseif length(varargin) == 4
        eps = varargin(1);
        m1 = varargin(2);
        tau = varargin(3);
        N_max = varargin(4);
        show = 10;
    elseif length(varargin) == 5
        eps = varargin(1);
        m1 = varargin(2);
        tau = varargin(3);
        N_max = varargin(4);
        show = varargin(5);
    end
    alg_params = {eps , m1 , tau , N_max , show};
    [X_train, X_val, Y_train, Y_val] = KFoldSplit(X, Y, K);
    mse_train = [];
    mse_val = [];
    for i = 1:K
        model = SVR(X_train{i}, Y_train{i}, model_params{:});
        alpha_opt = train_alg(@model.objfunc, @model.objgrad, model.A, model.b, model.A_eq, model.b_eq, model.alpha, alg_params{:});
        model.update_alpha(alpha_opt);
        Y_predict_train = model.predict(X_train{i});
        mse_train = [mse_train ; max([abs(Y_predict_train - Y_train{i}) - model_params{1} ; 0])];
        Y_predict_val = model.predict(X_val{i});
        mse_val = [mse_val ; max([abs(Y_predict_val - Y_val{i}) - model_params{1} ; 0])];
    end
    mean_mse_train = mean(mse_train);
    stdev_mse_train = std(mse_train);
    mean_mse_val = mean(mse_val);
    stdev_mse_val = std(mse_val);
    fprintf("Training completed, hyperparameters: %d\n", model_params);
    fprintf("mse on training set: %f +- %f\n", mean_mse_train, stdev_mse_train);
    fprintf("mse on validation set: %f +- %f\n", mean_mse_val, stdev_mse_val);
end