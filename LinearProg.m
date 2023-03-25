function x_min = LinearProg(c_vec, A, b, A_eq, b_eq)
    Table = [A eye(size(A, 1)) b ; A_eq zeros(size(A_eq, 1), size(A_eq, 2)) b_eq]; %eye = identity matrix with same size of A
    Table = [Table ; c_vec.' zeros(size(Table, 2) - size(c_vec, 1))];
    pivots_idx = [];
    while true
        [enter, idx_enter] = min(Table(end,:));
        if enter >= 0
            break;
        end
        rhs_div = Table(:end-1,end)/Table(:end-1,idx_enter);
        [exit, idx_exit] = min(rhs_div(rhs_div>0));
        if size(exit, 2) == 0
            break;
        end
        pivots_idx = [pivots_idx ; idx_enter idx_exit];
        Table(idx_exit,:) = Table(idx_exit,:)/Table(idx_exit,idx_enter);
        if idx_exit ~= 1
            Table(1:idx_exit-1,:) = Table(1:idx_exit-1,:) - Table(1:idx_exit-1,idx_enter)*Table(idx_exit,:);
        end
        Table(idx_exit+1:,:) = Table(idx_exit+1:,:) - Table(idx_exit+1:,idx_enter)*Table(idx_exit,:);
    end
    x_min = zeros(size(Table, 1) - 1, 1);
    x_min(pivots_idx(:,1)) = Table(pivots_idx(:,2),end);
    x_min = x_min(:size(c_vec, 1), 1);
end