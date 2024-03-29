function alpha = LineSearch(func, grad_func, x_origin, direction, alpha_init, m1, tau)
    alpha = alpha_init;
    while func(x_origin + alpha*direction) > func(x_origin) + m1*alpha*grad_func(x_origin).'*direction
        alpha = alpha*tau;
        if alpha < 1e-3
            alpha = 1e-3;
            break;
        end
    end
end