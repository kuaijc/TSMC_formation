matching();

function matching() 
    rng(42)
    global N;
    N = 5;      % agent num
    global NN;
    NN = 5;
    Ni = 2;     % degree
    
    % parameter selection
    alpha = 0.3;    % tunable step-size of PDC-ADMM
    rho = 2;    % tunable penalty force of PDC-ADMM
    matching_times = 300;

    % the undirected communication topogy is a ring graph
    Adjacency_Matrix = zeros(N, N);
    for i = 1:N
        for j = 1:1
            Adjacency_Matrix(i, mod(i + N - 1 - j, N) + 1) = 1;
            Adjacency_Matrix(i, mod(i + j - 1, N) + 1) = 1;
        end
    end
    IN = ones(1, N);    % row vector
    x = zeros(N, N);
    y = rand(N, N);
    lam = rand(1, N);
    psi = zeros(N, N);  
    global c;
    c = ones(N, N); % cost

    whole_x = [];
    unmatched = true;
    loop_times = 0;

    while unmatched
        loop_times = loop_times + 1;
        k1 = 1;
        unmatched = false;
        if loop_times ~= 1
            matching_times = 100;
        end
        while k1 <= matching_times
            vert_x = [];        % 1*(NN*NN)
            for i = 1:NN
                for j = 1:NN
                    vert_x = [vert_x, x(i, j)];
                end
            end
            whole_x = [whole_x; vert_x];   % times*(NN*NN)
    
            for i = 1:N
                sigma_y = zeros(1, N);
                for j = 1:N
                    if Adjacency_Matrix(i, j) == 1
                        sigma_y = sigma_y + y(i, :) + y(j, :);
                    end
                end
                w_xi = (IN / N - x(i, :) - psi(i, :) + rho * sigma_y) / (2 * rho * Ni);    % denote wi(xi_k)
                xi_hat = x(i, :) - alpha * (c(i, :) + (lam(i) + IN * x(i, :)' - 1) * IN - w_xi);    % equation 18a
                x(i, :) = min(1.0, max(0.0, xi_hat));     % equation 18b
                y(i, :) = (IN / N - x(i, :) - psi(i, :) + rho * sigma_y) / (2 * rho * Ni);    % equation 16   
                lam(i) = lam(i) + alpha * (IN * x(i, :)' - 1);  % equation 18c
            end
    
            for i = 1:N
                minus_y = zeros(1, N);
                for j = 1:N
                    if Adjacency_Matrix(i, j) == 1
                        minus_y = minus_y + y(i, :) - y(j, :);
                    end
                end
                psi(i, :) = psi(i, :) + rho * minus_y;  % equation 13
            end
    
            k1 = k1 + 1;
        end
        delta = 0.1 * rand();
        for i = 1:N
            perturb = false;
            for j = 1:N
                x(i, j) = round(x(i, j), 3);
                if x(i, j) > 0 && x(i, j) < 1
                    perturb = true;
                    unmatched = true;
                    break
                end
            end
            if perturb
                for j = 1:N
                    x(i, j) = round(x(i, j), 3);
                    if x(i, j) ~= 0
                        random_num = (-delta / N) * rand();
                        c(i, j) = c(i, j) + round(random_num, 2);
                    else
                        random_num = delta + (1-delta) * rand();
                        c(i, j) = c(i, j) + round(random_num, 2);
                    end
                end
            end
        end
    end

    single_xij = cell(NN * NN, 1);
    [num_rows, ~] = size(whole_x);
    
    % read all the elements for visualization
    for ii = 1:num_rows
        num = 1;
        for jj = 1:NN * NN
            single_xij{jj} = [single_xij{jj}, whole_x(ii, num)];
            num = num + 1;
        end
    end
    paint_each_element(single_xij, num_rows);
end

function paint_each_element(all_elements, real_times)
    global NN;
    x_value = 0:real_times - 1;

    hold on;
    for f = 1:(NN * NN)
        plot(x_value, cell2mat(all_elements(f,:)), 'LineWidth', 2);
    end
    xlabel('Iteration Times k', 'FontSize', 20, 'FontName', 'Times New Roman');
    ylabel('Value of Element \chi_{ij}', 'FontSize', 20, 'FontName', 'Times New Roman');
    xlim([0, x_value(end)]);
    ylim([-0.5, 1.5]);

    grid on;
    ax = gca;
    ax.FontSize = 16;
    ax.FontName = 'Times New Roman';
    ax.LineWidth = 1.2;    
%     set(gcf, 'Renderer', 'painters');   % ensure the fonttype is right
%     print(gcf, '-dpdf', '-painters', 'matching_result.pdf');
    hold off;
    shg;
end

