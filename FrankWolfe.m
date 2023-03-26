function x_min = FrankWolfe(func, grad_func, A, b, A_eq, b_eq, x_init, eps, tau, N_max)
    i = 0;
    x = x_init;
    while i < N_max
        grad = grad_func(x);
        x_bar = LinearProg(grad, A, b, A_eq, b_eq);
        direction = x_bar - x;
        if grad.'*direction > - eps
            break;
        end
        alpha = LineSearch(func, grad_func, x, direction, 1, eps, tau);
        x = x + alpha*direction;
        i = i + 1;
    end
    x_min = x;
end