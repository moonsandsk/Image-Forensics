classdef ImageTamperingDetector
    properties
        originalImage  % stores the double precision image
        grayImage      % grayscale version for analysis
        filename       % path to the file
    end

    methods
  
        function obj = ImageTamperingDetector(filename)
            obj.filename = filename;
            
            % load the specific test image
            % note: usually we'd use obj.filename here, but for this test we are hardcoding
            img = imread('FlowerInGrass_Test.jpg');
            
            % sometimes pngs have a 4th channel (transparency), we don't need that
            if size(img,3) == 4
                img = img(:,:,1:3);
            end
            
            % convert to double (0 to 1 range) for math operations
            obj.originalImage = im2double(img);
            
            % keep a grayscale copy since most algorithms work on intensity
            obj.grayImage = rgb2gray(obj.originalImage);
        end
        
        
        function [ela_map, ela_gray, stats] = detectELA(obj, quality)
            % error level analysis: save as jpeg and compare differences
            
            temp_jpeg = 'temp_ela_detection.jpg';
            
            % save the image with specific quality to induce compression
            imwrite(obj.originalImage, temp_jpeg, 'jpg', 'Quality', quality);
          
            % read it back to get the compressed version
            recomp_img = imread(temp_jpeg);
            
            % just in case dimensions shift slightly during save/load
            if size(recomp_img, 1) ~= size(obj.originalImage, 1) || ...
               size(recomp_img, 2) ~= size(obj.originalImage, 2)
                recomp_img = imresize(recomp_img, ...
                    [size(obj.originalImage, 1), size(obj.originalImage, 2)]);
            end
            recomp_norm = im2double(recomp_img);
            
            % find the absolute difference (the error)
            ela_map = abs(obj.originalImage - recomp_norm);
            
            % flatten to grayscale for visualization
            if size(ela_map, 3) == 3
                ela_gray = rgb2gray(ela_map);
            else
                ela_gray = ela_map;
            end
            
            % boost the contrast so we can actually see the error patterns
            ela_gray = imadjust(mat2gray(ela_gray));
            % adaptive equalization helps bring out faint details
            ela_gray = adapthisteq(ela_gray, 'ClipLimit', 0.02);
          
            % clean up the temp file
            delete(temp_jpeg);
            
            % calculate some basic stats to help us decide if it's fake
            stats.mean = mean(ela_gray(:));
            stats.std = std(ela_gray(:));
            stats.max = max(ela_gray(:));
            stats.min = min(ela_gray(:));
            
            % simple logic: anything way above the mean is suspicious
            adaptive_thresh = stats.mean + 2*stats.std;
            stats.suspicious_pixels = sum(ela_gray(:) > adaptive_thresh);
            stats.suspicion_percentage = ...
                (stats.suspicious_pixels / numel(ela_gray)) * 100;
        end
        
        
        function [artifact_map, suspicious_regions] = detectJPEGArtifacts(obj)
            % jpeg splits images into 8x8 blocks, tampering often breaks this grid
            
            block_size = 8;
            % work in ycbcr space, luminance (y) holds the artifacts
            img_yuv = rgb2ycbcr(im2uint8(obj.originalImage));
            Y = double(img_yuv(:,:,1));
            
            % look for blocks with unusual standard deviation
            fun = @(block_struct) std2(block_struct.data);
            std_map = blockproc(Y, [block_size block_size], fun);
            
            % also look for blocks with sharp internal edges
            fun2 = @(block_struct) mean2(abs(diff(block_struct.data)));
            edge_map = blockproc(Y, [block_size block_size], fun2);
            
            % combine the two evidence maps
            artifact_map = mat2gray(std_map) + mat2gray(edge_map);
            artifact_map = mat2gray(artifact_map);
            % smooth it out to reduce noise
            artifact_map = medfilt2(artifact_map, [3 3]);
          
            % thresholding to find the worst blocks
            threshold = graythresh(artifact_map);
            suspicious_regions = artifact_map > (threshold * 1.2);
        end
        
        
        function [noise_map, noise_clusters] = detectNoiseInconsistency(obj)
            % check if different parts of the image have different camera noise
            
            noise_maps = cell(3, 1);
            
            % try denoising with different kernel sizes to capture different noise frequencies
            wiener_img1 = wiener2(obj.grayImage, [3 3]);
            noise_maps{1} = abs(obj.grayImage - wiener_img1);
            
            wiener_img2 = wiener2(obj.grayImage, [5 5]);
            noise_maps{2} = abs(obj.grayImage - wiener_img2);
            
            wiener_img3 = wiener2(obj.grayImage, [7 7]);
            noise_maps{3} = abs(obj.grayImage - wiener_img3);
            
            % average them for a robust noise estimate
            noise_map = (noise_maps{1} + noise_maps{2} + noise_maps{3}) / 3;
            noise_map = mat2gray(noise_map);
            
            % use kmeans to separate the image into "background noise" and "alien noise"
            pixels = noise_map(:);
            % add tiny random noise to prevent kmeans from crashing on flat images
            pixels = pixels + randn(size(pixels)) * 0.001; 
            [idx, ~] = kmeans(pixels, 3, 'Replicates', 5, 'MaxIter', 200);
            noise_clusters = reshape(idx, size(noise_map));
        end
        
       
        function [edge_map, suspicious_edges, magnitude] = detectEdgeInconsistency(obj)
            % spliced objects often have edges that are too sharp or too blurry
            
            % calculate gradient magnitude
            [Gx, Gy] = imgradientxy(obj.grayImage, 'sobel');
            magnitude = sqrt(Gx.^2 + Gy.^2);
            
            % detect edges using multiple methods to get them all
            edge_canny = edge(obj.grayImage, 'Canny');
            edge_sobel = edge(obj.grayImage, 'Sobel');
            edge_prewitt = edge(obj.grayImage, 'Prewitt');
            
            % combine them
            edge_map = edge_canny | edge_sobel | edge_prewitt;
            
            % get the strength of just the edge pixels
            edge_magnitude = magnitude(edge_map);
            mean_mag = median(edge_magnitude); % median is safer than mean here
            std_mag = std(edge_magnitude);
            
            % identify edges that are way stronger/weaker than normal
            suspicious_edges = edge_map & (abs(magnitude - mean_mag) > 2*std_mag);
            
            % remove tiny stray pixels
            suspicious_edges = bwareaopen(suspicious_edges, 5);
        end
        
        
        function [copy_move_map, matched_blocks] = detectCopyMove(obj)
            % this part looks for identical patches (cloning)
            % it's computationally heavy so we step by 4 pixels
            
            patch_size = 16;
            step = 4; 
            img = im2uint8(obj.grayImage);
            [h, w] = size(img);
            
            patches = {};
            coords = [];
            idx = 1;
            
            % break image into blocks
            for y = 1:step:(h-patch_size+1)
                for x = 1:step:(w-patch_size+1)
                    patch = img(y:y+patch_size-1, x:x+patch_size-1);
                    
                    % use dct features (more robust than raw pixels)
                    dct_patch = dct2(double(patch));
                    features = dct_patch(1:8, 1:8); % take low freq coefficients
                    
                    patches{idx} = features(:)';
                    coords(idx,:) = [y x];
                    idx = idx + 1;
                end
            end
            
            % organize data for correlation
            data = cell2mat(patches');
            data = double(data);
            
            % z-score normalization makes it lighting invariant
            data_mean = mean(data, 2);
            data_std = std(data, [], 2);
            data_std(data_std < 1e-10) = 1; % avoid divide by zero
            data = (data - data_mean) ./ data_std;
            
            % calculate similarity matrix (heavy math part)
            C = data * data' / size(data, 2);
            
            % determine dynamic threshold based on data distribution
            corr_values = C(triu(true(size(C)), 1));
            threshold = mean(corr_values) + 3*std(corr_values);
            threshold = max(threshold, 0.95); % don't go too low
            
            n = size(C, 1);
            copy_move_map = zeros(h, w);
            matched_blocks = 0;
            
            for i = 1:n
                for j = i+1:n
                    if C(i,j) > threshold
                        y1 = coords(i,1); x1 = coords(i,2);
                        y2 = coords(j,1); x2 = coords(j,2);
                        
                        % check distance - close blocks are usually just texture
                        % far blocks are likely clones
                        dist = sqrt((y1-y2)^2 + (x1-x2)^2);
                        if dist > patch_size*2
                            copy_move_map(y1:y1+patch_size-1, x1:x1+patch_size-1) = 1;
                            copy_move_map(y2:y2+patch_size-1, x2:x2+patch_size-1) = 1;
                            matched_blocks = matched_blocks + 1;
                        end
                    end
                end
            end
            
            copy_move_map = logical(copy_move_map);
            
            % fill in gaps to make solid shapes
            copy_move_map = imclose(copy_move_map, strel('disk', 3));
            copy_move_map = bwareaopen(copy_move_map, 100);
        end
        
        
        function report = generateReport(obj)
            % runs everything and builds the final dashboard
            
            quality = 95; 
            
            % run all the detectors
            [report.ela_map, report.ela_gray, report.ela_stats] = obj.detectELA(quality);
            [report.artifact_map, report.suspicious_regions] = obj.detectJPEGArtifacts();
            [report.noise_map, report.noise_clusters] = obj.detectNoiseInconsistency();
            [report.edge_map, report.suspicious_edges, report.edge_magnitude] = obj.detectEdgeInconsistency();
            [report.copy_move_map, report.matched_blocks] = obj.detectCopyMove();
            
            % resize everything to match so we can add them up
            sz = size(report.ela_gray);
            artifact_resized = imresize(report.artifact_map, sz);
            noise_resized = imresize(report.noise_map, sz);
            edge_resized = imresize(double(report.suspicious_edges), sz);
            copy_move_resized = imresize(double(report.copy_move_map), sz);
            
            % fuse evidence with weights (ela is usually strongest)
            weights = [0.3, 0.2, 0.2, 0.15, 0.15];
            combined = weights(1)*mat2gray(report.ela_gray) + ...
                      weights(2)*artifact_resized + ...
                      weights(3)*noise_resized + ...
                      weights(4)*edge_resized + ...
                      weights(5)*copy_move_resized;
            
            report.combined_suspicion = mat2gray(combined);
            
            % smooth the final heatmap
            report.combined_suspicion = medfilt2(report.combined_suspicion, [5 5]);
      
            % create the visualization figure
            figure('Name', 'Image Tampering Detection Report', 'Position', [100 100 1200 600]);
            subplot(3,4,1), imshow(obj.originalImage), title('Original Image');
            subplot(3,4,2), imhist(report.ela_gray), title('ELA Histogram');
            subplot(3,4,3), imshow(imadjust(report.ela_gray)), title('ELA Enhanced');
            subplot(3,4,4), imshow(report.artifact_map), title('JPEG Artifacts');
            subplot(3,4,5), imshow(report.noise_map), title('Noise Map');
            subplot(3,4,6), imshow(report.edge_map), title('Edge Map');
            subplot(3,4,7), imshow(report.suspicious_edges), title('Suspicious Edges');
            
            % threshold the combined map to show "guilty" areas
            thresh = graythresh(report.combined_suspicion);
            overlay = imoverlay(obj.originalImage, report.combined_suspicion > thresh*1.5, [1 0 0]);
            subplot(3,4,10), imshow(overlay), title('Tampering Overlay');
            subplot(3,4,11), imshow(report.combined_suspicion), title('Combined Suspicion');
            
            % simple rule-based verdict
            confidence = report.ela_stats.suspicion_percentage;
            if confidence > 15
                verdict = 'High Probability of Tampering';
            elseif confidence > 5
                verdict = 'Moderate Suspicion';
            else
                verdict = 'Low Suspicion';
            end
            subplot(3,4,12), axis off, 
            text(0.5, 0.5, {verdict, sprintf('Confidence: %.2f%%', confidence)}, ...
                'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
        end
    end
end

% helper function to paint red pixels on top of the image
function out = imoverlay(img, mask, color)
    if size(img,3) == 1
        img = repmat(img, [1,1,3]);
    end
    out = img;
    for c = 1:3
        channel = img(:,:,c);
        channel(mask) = color(c);
        out(:,:,c) = channel;
    end
end