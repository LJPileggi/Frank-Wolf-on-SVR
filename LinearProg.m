function x_min = LinearProg(c_vec, A, b, A_eq, b_eq)
    Table = [A eye(size(A, 1)) b];
    if size(A_eq,1) ~= 0 && size(b_eq,1) ~= 0
        Table = [Table ; A_eq zeros(1, size(A_eq, 1), size(A_eq, 2)) b_eq];
    end
    Table = [Table ; c_vec.' zeros(1, size(Table, 2) - size(c_vec, 1))];
    pivots_idx = [];
    while true
        [enter, idx_enter] = min(Table(end,:));
        if enter >= 0
            break;
        end
        rhs_div = Table(1:end-1,end)./Table(1:end-1,idx_enter);
        exit = min(rhs_div(rhs_div>0));
        if exit == inf
            break;
        end
        idx_exit = find(rhs_div==exit);
        if size(exit, 2) == 0
            break;
        end
        pivots_idx = [pivots_idx ; idx_enter idx_exit];
        Table(idx_exit,:) = Table(idx_exit,:)/Table(idx_exit,idx_enter);
        if idx_exit ~= 1
            Table(1:idx_exit-1,:) = Table(1:idx_exit-1,:) - Table(1:idx_exit-1,idx_enter)*Table(idx_exit,:);
        end
        Table(idx_exit+1:end,:) = Table(idx_exit+1:end,:) - Table(idx_exit+1:end,idx_enter)*Table(idx_exit,:);
    end
    x_min = zeros(size(Table, 1) - 1, 1);
    if size(pivots_idx, 1) ~= 0
        x_min(pivots_idx(:,1)) = Table(pivots_idx(:,2),end);
    end
    x_min = x_min(1:size(c_vec, 1), 1);
end