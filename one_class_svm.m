function [alpha, b] = one_class_svm(train_data, nu, kernel_function)
    % kernel_function is a function handle (e.g., @(x, y) x * y')
    
    n = size(train_data, 1);
    H = zeros(n, n);
    
    % Compute the kernel matrix
    for i = 1:n
        for j = 1:n
            H(i, j) = kernel_function(train_data(i, :), train_data(j, :));
        end
    end
    
    % Define the quadratic programming problem
    f = -ones(n, 1);
    Aeq = ones(1, n);
    beq = 1;
    lb = zeros(n, 1);
    ub = ones(n, 1) * (1 / (n * nu));
    
    % Solve the quadratic programming problem
    alpha = quadprog(H, f, [], [], Aeq, beq, lb, ub);
    
    % Find support vectors
    support_vector_indices = find(alpha > 1e-6);
    
    % Compute the bias term
    b = 0;
    for i = 1:length(support_vector_indices)
        k = kernel_function(train_data(support_vector_indices(i), :), train_data);
        b = b + (alpha' * k' - 1);
    end
    b = b / length(support_vector_indices);
end
