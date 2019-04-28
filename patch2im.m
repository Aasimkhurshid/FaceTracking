function [img] = patch2im(patch, size_img, size_patch, size_skip, border)
if(nargin < 4), size_skip = [3 3]; end;
if(nargin < 5), border = 'off'; end
img = zeros(size_img);
w = zeros(size_img);
patch_loc = patchLocation(size_img, size_patch, size_skip, border);
for n=1:size(patch_loc,3)
    img(patch_loc(:,:,n)) = img(patch_loc(:,:,n)) + reshape(patch(:,n), size_patch);
    w(patch_loc(:,:,n)) = w(patch_loc(:,:,n)) + 1;
end
img = img ./ w;