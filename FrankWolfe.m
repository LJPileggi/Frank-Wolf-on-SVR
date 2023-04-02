function x_min = FrankWolfe(func, grad_func, A, b, A_eq, b_eq, x_init, eps, m1, tau, N_max, show)
    i = 0;
    x = x_init;
    while i < N_max
        grad = grad_func(x);
        x_bar = LinearProg(grad, A, b, A_eq, b_eq);
        direction = x_bar - x;
        contr = grad.'*direction;
        if mod(i, show) == 0
            fprintf("iteration %i, grad: %f\n", i, contr);
        end
        if contr > - eps
            break;
        end
        alpha = LineSearch(func, grad_func, x, direction, 1, m1, tau);
        x = x + alpha*direction;
        i = i + 1;
    end
    x_min = x;
end