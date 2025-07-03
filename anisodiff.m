function diff_im = anisodiff(im, num_iter, delta_t, kappa, option)
% ANISODIFF - Anisotropic diffusion filtering (Perona & Malik)
%
% im        - Input image (grayscale)
% num_iter  - Number of iterations
% delta_t   - Integration constant (0 <= delta_t <= 1/7 for stability)
% kappa     - Gradient modulus threshold controlling conduction
% option    - 1 for Perona-Malik (exponential), 2 for Perona-Malik (quadratic)

im = double(im);
diff_im = im;
[rows, cols] = size(im);

% Define 2D convolution masks for gradient calculation
dx = [1 -1];
dy = [1; -1];

for t = 1:num_iter
    % Compute gradients in four directions
    nablaN = conv2(diff_im, dy, 'same');
    nablaS = -nablaN;
    nablaE = conv2(diff_im, dx, 'same');
    nablaW = -nablaE;

    % Compute conduction coefficients
    if option == 1
        cN = exp(-(nablaN/kappa).^2);
        cS = exp(-(nablaS/kappa).^2);
        cE = exp(-(nablaE/kappa).^2);
        cW = exp(-(nablaW/kappa).^2);
    elseif option == 2
        cN = 1 ./ (1 + (nablaN/kappa).^2);
        cS = 1 ./ (1 + (nablaS/kappa).^2);
        cE = 1 ./ (1 + (nablaE/kappa).^2);
        cW = 1 ./ (1 + (nablaW/kappa).^2);
    else
        error('Option must be 1 (exponential) or 2 (quadratic).');
    end

    % Update image using diffusion equation
    diff_im = diff_im + delta_t * (cN.*nablaN + cS.*nablaS + cE.*nablaE + cW.*nablaW);
end

diff_im = uint8(diff_im); % Convert back to uint8 format
end
