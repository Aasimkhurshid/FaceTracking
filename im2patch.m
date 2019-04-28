function [patch] = im2patch(img, size_patch, size_skip, border)
% im2patch converts [Y X] size image
% first to [size_patch(1) size_patch(2) num_patch] size 3D array,
% then to [size_patch(1)*size_patch(2) num_patch] size 2D array.
if(nargin < 3), size_skip = [3 3]; end
if(nargin < 4), border = 'off'; end
patch_loc = patchLocation(size(img), size_patch, size_skip, border);
patch = img(patch_loc);
patch = reshape(patch, [prod(size_patch) size(patch_loc,3)]);