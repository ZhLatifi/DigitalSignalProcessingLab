%% 1-4-a)

pic = imread("lena.bmp");
imshow(pic);
title("Original Image");
%% 1-4-b)

pic_d = im2double(pic);
imshow(pic_d);
title("Original Image - Double");
%% 1-4-c)

imhist(pic_d);
title("Histogram of Original Image");
%% 1-4-d)

J = histeq(pic_d, 2);
imshowpair(pic_d, J,'montage');
title("Original Vs. after using histeq() Image");
%% 1-4-e)

imhist(J);
title("after using histeq() Histogram");
%% 
%% 2-4-a)

pic = im2double(imread("Image02.jpg"));
imshow(pic);
title("Original Image");
%% 
%% 2-4-b)

pic_n = imnoise(rgb2gray(pic), 'gaussian', 0, 0.04);
imshow(pic_n);
title("Noisy Image");
%% 
%% 2-4-c)

mean3 = (1/9)*ones(3,3);
pic_dn3 = imfilter(pic_n , mean3);
imshow(pic_dn3);
title("after Using Mean Filter - 3*3");
%% 2-4-d)

mean5 = (1/25)*ones(5,5);
pic_dn5 = imfilter(pic_n , mean5);
imshow(pic_dn5);
title("after Using Mean Filter - 5*5");
%% 2-4-e)

pic_n2 = imnoise(rgb2gray(pic), "salt & pepper", 0.1);
imshow(pic_n2);
title("Noisy Image - S&P");
%% 2-4-f)

mean3 = (1/9)*ones(3,3);
pic_dn2 = imfilter(pic_n2 , mean3);
imshow(pic_dn2);
title("after Using Mean Filter - 3*3");
%% 2-4-g)

freqz(Num);
title("FDA Tool filter");
filt2 = ftrans2(Num);
freqz2(filt2);
title("FDA Tool filter - 2D");
%% 2-4-h)

pic_dnfG = imfilter(pic_n, filt2);
imshow(pic_dnfG);
title("G-noisy Image after using FDA filter");
pic_dnfSP = imfilter(pic_n2, filt2);
imshow(pic_dnfSP);
title("S&P noisy Image after using FDA filter");
%% 2-4-i,j,k)

medfilt = medfilt2(pic_n2);
imshow(medfilt);
title("after Using Median Filter");
%% 3-4-a)

pic = im2double(imread("square2.jpg"));
imshow(pic);
title("Original Image");
[cA, cH, cV, cD] = dwt2(pic, 'db1');
imagesc(cA);
title("cA");
imagesc(cD);
title("cD");
%% 3-4-b)

imagesc(cH);
title("cH");
imagesc(cV);
title("cV");
imshow(idwt2(cA, 1000*cH, cV, cD, 'db1'));
title("Highlighted Horizontal lines");
%% 4-4-a)

pic = im2double(imread("Image04.png"));
imshow(pic);
title("Original Image");
motion = fspecial('motion', 15, 20);
blurred = imfilter(pic, motion, 'circular');
imshow(blurred);
title("Blurred Image");
%% 4-4-b)

wnr = deconvwnr(blurred, motion, 0.001);
imshow(wnr);
title('Restored Blurred Image');
%% 4-4-c)

blurred_n = imnoise(rgb2gray(blurred), 'gaussian', 0, 0.01);
imshow(blurred_n);
title("Blurred Noisy Image");
%% 4-4-d)

estimated_nsr = 0.01 / var(im2double(pic(:)));
wnr2 = deconvwnr(blurred_n, motion, estimated_nsr);
imshow(wnr2);
title('Restoration of Blurred Noisy Image');
%% 4-5-a)

pic = im2double(imread("glass.tif"));
imshow(pic);
title("Original Image");
%% 4-5-b)

pic_fft2 = fft2(pic);
mesh(abs(fftshift(pic_fft2)));
figure();
mesh(angle(fftshift(pic_fft2)));
%% 4-5-c)

% function Output_image = FFT_LP_2D(input_image, cutoff_frequency)
%     
%     fft_image = fft2(input_image);
%     fft_shifted = fftshift(fft_image);
%     
%     [rows, cols] = size(input_image);
%     center_row = rows / 2;
%     center_col = cols / 2;
%     
%     % low-pass filter
%     filter_mask = zeros(rows, cols);
%     for i = 1:rows
%         for j = 1:cols
%             distance = sqrt((i - center_row)^2 + (j - center_col)^2);
%             if ((distance/center_row)*pi) <= cutoff_frequency
%                 filter_mask(i, j) = 1;
%             end
%         end
%     end
%     
%     
%     filtered_fft = fft_shifted .* filter_mask;
%     ifft_shifted = ifftshift(filtered_fft);
%     
%     % 2D IFFT
%     Output_image = ifft2(ifft_shifted, 'symmetric');
% end
%% 4-5-d)

result = FFT_LP_2D(pic, 0.1*pi);
imshow(result);
title("Filtered Image - LP-2D");
%% 4-5-e)

dwn_pic = downsample(downsample(pic, 4)', 4)';
figure();
subplot(1,2,1);
imshow(dwn_pic);
title("Downsampled Image");

dwn_pic_fil = FFT_LP_2D(dwn_pic, 0.5*pi);
subplot(1,2,2);
imshow(dwn_pic_fil);
title("Filtered, w = 0.5*pi");
