close all, clear all
  
% load xbox;
xbox = imread('split1.png');
% https://mathworks.com/help/wavelet/ref/haart2.html
[a,h,v,d] = haart2(xbox, 1);

for a = {xbox, h, v, d};  % Store the images in a cell array
  mode(a{1})
  figure();
  colormap('gray');
  imagesc(a{1});
  axis off;
end
