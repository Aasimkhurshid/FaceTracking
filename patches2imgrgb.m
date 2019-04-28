function img = patches2imgrgb(patches_mat,patch_size,pixels_hor,pixels_ver,dif_ver)

    % Usage example:
    
    % img = imread(...);
    % patch_size = 7;
    % [X,dif_ver] = imgrgb2patches(img,patch_size);
    % img_rec = patches2imgrgb(X,patch_size,size(img,1),size(img,2),dif_ver);

    img = zeros(pixels_hor,pixels_ver,3,'uint8');
    patches_ver = (pixels_ver - dif_ver)/patch_size;

    for k = 1:size(patches_mat,2)
        i = floor((k-1)/patches_ver);
        j = mod(k-1,patches_ver);
        patch = reshape(patches_mat(:,k),patch_size,patch_size,3);
        img(i*patch_size+1:(i+1)*patch_size,j*patch_size+1:(j+1)*patch_size,:) = patch;
    end
    
    img(pixels_hor-patch_size+1:end,:,:) = mode(mode(img(1:20,1:20,1)));
    img(:,pixels_ver-patch_size+1:end,:) = mode(mode(img(1:20,1:20,1)));
   
end

