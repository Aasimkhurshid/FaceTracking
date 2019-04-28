function [patches_mat,dif_ver] = imgrgb2patches(img,patch_size)

    % Usage example:
    
    % img = imread(...);
    % patch_size = 7;
    % [X,dif_ver] = imgrgb2patches(img,patch_size);
    % img_rec = patches2imgrgb(X,patch_size,size(img,1),size(img,2),dif_ver);

    pixels_hor = size(img,1);
    pixels_ver = size(img,2);

    patches_hor = floor(pixels_hor/patch_size);
    patches_ver = floor(pixels_ver/patch_size);
    patches_total = patches_hor*patches_ver;

    dif_ver = pixels_ver - patches_ver*patch_size;

    patch_dim = patch_size*patch_size*3;
    patches_mat = zeros(patch_dim,patches_total);
    k = 1;
    for i = 1:patches_hor
        for j = 1:patches_ver
            patch = img((i-1)*patch_size+1:i*patch_size,(j-1)*patch_size+1:j*patch_size,:);
            patches_mat(:,k) = reshape(patch,patch_dim,1);
            k = k + 1;
        end
    end

end

