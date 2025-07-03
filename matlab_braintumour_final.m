clc; clear; close all;
[I, path] = uigetfile({'*jpg', 'Select an input image'});
if isequal(I, 0)
    disp('No file selected');
    return;
end
str = fullfile(path, I);
s = imread(str);

% Start timer for performance evaluation
tic;

% Display input image
figure;
imshow(s);
title('Input Image', 'FontSize', 20);

% Convert to grayscale if needed
if size(s, 3) > 1
    s = rgb2gray(s);
end

% Wavelet Transform for Feature Extraction
[LL, LH, HL, HH] = dwt2(s, 'haar');

% Gabor Filter for Texture Enhancement
gaborArray = gabor(2, [0 45 90 135]);
gaborMag = imgaborfilt(s, gaborArray);

% Apply Anisotropic Diffusion Filtering
num_iter = 10;
delta_t = 1/7;
kappa = 15;
option = 2;
filtered_img = anisodiff(s, num_iter, delta_t, kappa, option);
filtered_img = uint8(filtered_img);
filtered_img = imresize(filtered_img, [256, 256]);

% Adaptive Thresholding for Better Segmentation
thresh_img = adaptthresh(filtered_img, 0.5);
bw_img = imbinarize(filtered_img, thresh_img);

% Label connected components
label = bwlabel(bw_img);
stats = regionprops(logical(bw_img), 'Solidity', 'Area', 'BoundingBox');

% Find high-density areas
density = [stats.Solidity];
area = [stats.Area];

if isempty(area)
    msgbox('No Tumor Detected!', 'Status');
    return;
end

high_dense_area = density > 0.6;
max_area = max(area(high_dense_area));

% Find the tumor
tumor_label = find(area == max_area, 1);  % Ensure a valid index
tumor = ismember(label, tumor_label);

if max_area > 100
    figure;
    imshow(tumor);
    title('Tumor Alone', 'FontSize', 20);
else
    msgbox('No Tumor Detected!', 'Status');
    return;
end

% Bounding box
box = stats(tumor_label);
wantedBox = box.BoundingBox;

figure;
imshow(filtered_img);
title('Bounding Box', 'FontSize', 20);
hold on;
rectangle('Position', wantedBox, 'EdgeColor', 'y');
hold off;

% Performance evaluation
latency = toc;
throughput = 1 / latency;

% Ensure 'tumor' is the same size as 's'
tumor = imresize(tumor, size(s));

% Extract tumor pixels safely
tumor_pixels = s(tumor(:));

% Statistical Features
mean_intensity = mean(tumor_pixels, 'omitnan');
std_intensity = std(double(tumor_pixels), 'omitnan');
valid_pixels = tumor_pixels(~isnan(tumor_pixels)); % Remove NaN values
skewness_intensity = skewness(double(valid_pixels), 0); % Normalize by N

valid_pixels = tumor_pixels(~isnan(tumor_pixels)); % Remove NaN values
kurtosis_intensity = kurtosis(double(valid_pixels), 0); % Normalize by N


% Compute Fractal Dimension (Complexity of Tumor Shape)
N = sum(tumor(:));
r = 2; % Box size
fractal_dimension = log(N) / log(1/r);

% Compute Dice Similarity and Jaccard Index
ground_truth = imbinarize(filtered_img);

% Ensure size compatibility
tumor = imresize(tumor, size(ground_truth));

intersection = sum(sum(tumor & ground_truth));
union = sum(sum(tumor | ground_truth));

dice_score = 2 * intersection / (sum(tumor(:)) + sum(ground_truth(:)));
jaccard_index = intersection / union;

% Tumor Severity Classification
if max_area < 500
    severity = 'Low';
elseif max_area < 2000
    severity = 'Moderate';
else
    severity = 'Severe';
end

% Generate report as a text file
fileID = fopen('tumor_report.txt', 'w');
fprintf(fileID, 'Tumor Detection Report\n');
fprintf(fileID, '========================\n');
fprintf(fileID, 'Tumor Severity: %s\n', severity);
fprintf(fileID, 'Tumor Area: %.2f pixels\n', max_area);
fprintf(fileID, 'Throughput: %.4f images/sec\n', throughput);
fprintf(fileID, 'Latency: %.4f seconds\n', latency);
fprintf(fileID, 'Dice Similarity: %.4f\n', dice_score);
fprintf(fileID, 'Jaccard Index: %.4f\n', jaccard_index);
fprintf(fileID, 'Mean Intensity: %.4f\n', mean_intensity);
fprintf(fileID, 'Std Dev Intensity: %.4f\n', std_intensity);
fprintf(fileID, 'Skewness: %.4f\n', skewness_intensity);
fprintf(fileID, 'Kurtosis: %.4f\n', kurtosis_intensity);
fprintf(fileID, 'Fractal Dimension: %.4f\n', fractal_dimension);
fclose(fileID);

disp('Tumor report generated: tumor_report.txt');

% Histogram Analysis
figure;
imhist(s);
title('Histogram of Intensity Levels');

% 3D Visualization of Tumor
figure(5);
clf;

tumor_intensity = double(tumor) * 255;
[rows, cols] = size(tumor_intensity);
[x, y] = meshgrid(1:cols, 1:rows);

if sum(tumor(:)) == 0
    disp('⚠️ No tumor detected, skipping 3D visualization.');
else
    tumor_intensity(~tumor) = NaN;
    surf(x, y, tumor_intensity, 'EdgeColor', 'none');
    colormap(jet);
    shading interp;
    title('3D Tumor Visualization', 'FontSize', 20);
    xlabel('X-axis'); ylabel('Y-axis'); zlabel('Intensity');
    view(3);
    axis tight;
    colorbar;
end

disp('Press any key to close the figure and continue execution...');
pause;

% Final visualization
figure;
subplot(231); imshow(s); title('Input Image', 'FontSize', 20);
subplot(232); imshow(filtered_img); title('Filtered Image', 'FontSize', 20);
subplot(233); imshow(filtered_img); title('Bounding Box', 'FontSize', 20);
hold on; rectangle('Position', wantedBox, 'EdgeColor', 'y'); hold off;
subplot(234); imshow(tumor); title('Tumor Alone', 'FontSize', 20);
subplot(235); imshow(tumor - imerode(tumor, ones(3))); title('Tumor Outline', 'FontSize', 20);
subplot(236); imshow(filtered_img); title('Detected Tumor', 'FontSize', 20);

import mlreportgen.dom.*;
import mlreportgen.report.*;

% Create a PDF Report
report = Document('Brain_Tumor_Report', 'pdf');

% 🎯 **Title Page**
append(report, Heading(1, 'Brain Tumor Detection Report'));
append(report, Paragraph('Generated using MATLAB'));
append(report, HorizontalRule());

% 🎯 **Tumor Severity Section**
append(report, Heading(2, 'Tumor Severity'));
severityPara = Paragraph(['Severity Level: ', severity]);
severityPara.Bold = true;
append(report, severityPara);

% 🎯 **Tumor Analysis Data (Formatted Table)**
append(report, Heading(2, 'Tumor Analysis Results'));
tableData = {...  
    'Tumor Area (pixels)', max_area;
    'Mean Intensity', mean_intensity;
    'Standard Deviation', std_intensity;
    'Skewness', skewness_intensity;
    'Kurtosis', kurtosis_intensity;
    'Fractal Dimension', fractal_dimension;
    'Dice Similarity', dice_score;
    'Jaccard Index', jaccard_index;
    'Throughput (images/sec)', throughput;
    'Latency (seconds)', latency;
};
table = Table(tableData);
table.Style = {Border('solid'), Width('100%')};
append(report, table);

% 🎯 **Add Images to PDF (Input Image, Tumor Segmentation, Bounding Box, 3D Plot, Histogram)**

% 📌 **Save and Insert Input Image**
inputFile = 'input_image.png';
imshow(s);
saveas(gcf, inputFile);
close;
img1 = Image(inputFile);
img1.Width = '4in'; img1.Height = '3in';
append(report, Heading(2, 'Input Image'));
append(report, img1);

% 📌 **Save and Insert Tumor Detection Image**
tumorFile = 'tumor.png';
imshow(tumor);
saveas(gcf, tumorFile);
close;
tumorImg = Image(tumorFile);
tumorImg.Width = '4in'; tumorImg.Height = '3in';
append(report, Heading(2, 'Detected Tumor'));
append(report, tumorImg);

% 📌 **Save and Insert Bounding Box Image**
boxFile = 'bounding_box.png';
imshow(filtered_img);
hold on; rectangle('Position', wantedBox, 'EdgeColor', 'y'); hold off;
saveas(gcf, boxFile);
close;
boxImg = Image(boxFile);
boxImg.Width = '4in'; boxImg.Height = '3in';
append(report, Heading(2, 'Bounding Box'));
append(report, boxImg);

% 📌 **Save and Insert Histogram**
histFile = 'histogram.png';
imhist(s);
saveas(gcf, histFile);
close;
histImg = Image(histFile);
histImg.Width = '4in'; histImg.Height = '3in';
append(report, Heading(2, 'Histogram of Intensity Levels'));
append(report, histImg);

% 📌 **Save and Insert 3D Tumor Visualization**
if sum(tumor(:)) > 0
    figure; 
    [rows, cols] = size(tumor);
    [x, y] = meshgrid(1:cols, 1:rows);
    tumor_intensity = double(tumor) * 255;
    tumor_intensity(~tumor) = NaN; % Make non-tumor pixels transparent
    surf(x, y, tumor_intensity, 'EdgeColor', 'none');
    colormap(jet); shading interp;
    title('3D Tumor Visualization');
    xlabel('X-axis'); ylabel('Y-axis'); zlabel('Intensity');
    view(3); axis tight; colorbar;
    tumor3DFile = 'tumor_3D.png';
    saveas(gcf, tumor3DFile);
    close;
    
    % Add 3D Tumor Visualization to Report
    tumor3DImg = Image(tumor3DFile);
    tumor3DImg.Width = '4in'; tumor3DImg.Height = '3in';
    append(report, Heading(2, '3D Tumor Visualization'));
    append(report, tumor3DImg);
end

% 🎯 **Save & Open Report**
close(report);
rptview('Brain_Tumor_Report', 'pdf');
disp('📄 Brain Tumor Detection Report Generated Successfully!');
