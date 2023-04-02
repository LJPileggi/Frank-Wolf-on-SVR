function [X_train, X_val, Y_train, Y_val] = KFoldSplit(X, Y, K)
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