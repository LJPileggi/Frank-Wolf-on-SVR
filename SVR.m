classdef SVR
    properties
        w
        X
        b
        Y
        slack
        C
        eps
        init
    end
    methods
        function obj = SVR(X, Y, C, eps, init)
            obj.X = X;
            obj.Y = Y;
            obj.C = C;
            obj.eps = eps;
            obj.init = init;
            if init == "unif"
                obj.w = -1+2*rand(1, size(X, 1));
            elseif init == "zero"
                obj.w = zeros(1, size(X, 1));
            elseif init == "norm"
                obj.w = normrnd(0, 0.5, [1, size(X, 1)]);
            else
                error("Invalid initialisation method.")
            
            end
        end
        function objval = objfunc(obj)
            objval = dot(obj.w, obj.w)/2 + obj.C*sum(obj.slack);
        end
        function grad = objgrad(obj)
            grad = cat(0, obj.w, obj.C*ones(1, size(obj.X,1)), 0);
        end
    end
end