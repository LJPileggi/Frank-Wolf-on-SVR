classdef SVR
    properties
        alpha1
        alpha2
        alpha
        gamma
        X
        Y
        N
        Ker
        q
        Q
        A
        b
        C
        sigma
        eps
        init
    end
    methods
        function obj = SVR(X, Y, C, eps, sigma, init)
            obj.X = X;
            obj.Y = Y;
            obj.C = C;
            obj.N = size(obj.X, 2);
            obj.eps = eps;
            obj.sigma = sigma;
            obj.set_ker(obj);
            obj.q = cat([obj.Y - obj.eps], [-obj.Y - obj.eps]);
            obj.Q = [-obj.Ker obj.Ker ; obj.Ker -obj.Ker];
            obj.A = [-eye(obj.N), zeros(obj.N) ; zeros(obj.N) -eye(obj.N) ; eye(obj.N) zeros(obj.N) ; zeros(obj.N) eye(obj.N) ; ones(1, obj.N) - ones(1, obj.N) ; -ones(1, obj.N) ones(1, obj.N)];
            obj.b = [zeros(2*obj.N, 1) ; obj.C*ones(2*obj.N, 1) ; 0 ; 0];
            obj.init = init;
            if init == "unif"
                obj.alpha1 = -1+2*rand(size(X, 1), 1);
                obj.alpha2 = -1+2*rand(size(X, 1), 1);
                obj.update_alpha_gamma(obj);
            elseif init == "zero"
                obj.alpha1 = zeros(size(X, 1), 1);
                obj.alpha2 = zeros(size(X, 1), 1);
                obj.update_alpha_gamma(obj);
            elseif init == "norm"
                obj.alpha1 = normrnd(0, 0.5, [size(X, 1), 1]);
                obj.alpha2 = normrnd(0, 0.5, [size(X, 1), 1]);
                obj.update_alpha_gamma(obj);
            else
                error("Invalid initialisation method.");
            end
        end
        function obj = update_alpha_gamma(obj)
            obj.alpha = cat(1, obj.alpha1, obj.alpha2);
            obj.gamma = obj.alpha1 - obj.alpha2;
        end
        function obj = set_ker(obj)
            obj.Ker = exp(-(obj.X.' * obj.X)/(2*obj.sigma^2));
        end
        function objval = objfunc(alpha)
            objval = dot(obj.q, alpha) + alpha.'*obj.Q*alpha/2;
        end
        function grad = objgrad(alpha)
            grad = obj.q + obj.Q*alpha;
        end
        function predict = eval_predict(x)
            predict = dot(obj.gamma, exp(-(obj.X.' * x)/(2*obj.sigma^2)));
        end
    end
end