function im_vectors = computeDeblurMLEM(im_vectors, options, iter, subsets, gaussK, Nx, Ny, Nz)
%COMPUTEDEBLUR Computes the PSF deblurring phase for all selected MLEM
%algorithms

if options.mlem
    im_vectors.MLEM(:, iter + 1) = deblur(im_vectors.MLEM(:, iter + 1), options, iter, subsets, gaussK, Nx, Ny, Nz);
elseif options.MRP && options.OSL_MLEM
    im_vectors.MRP_MLEM(:, iter + 1) = deblur(im_vectors.MRP_MLEM(:, iter + 1), options, iter, subsets, gaussK, Nx, Ny, Nz);
elseif options.quad && options.OSL_MLEM
    im_vectors.Quad_MLEM(:, iter + 1) = deblur(im_vectors.Quad_MLEM(:, iter + 1), options, iter, subsets, gaussK, Nx, Ny, Nz);
elseif options.Huber && options.OSL_MLEM
    im_vectors.Huber_MLEM(:, iter + 1) = deblur(im_vectors.Huber_MLEM(:, iter + 1), options, iter, subsets, gaussK, Nx, Ny, Nz);
elseif options.L && options.OSL_MLEM
    im_vectors.L_MLEM(:, iter + 1) = deblur(im_vectors.L_MLEM(:, iter + 1), options, iter, subsets, gaussK, Nx, Ny, Nz);
elseif options.FMH && options.OSL_MLEM
    im_vectors.FMH_MLEM(:, iter + 1) = deblur(im_vectors.FMH_MLEM(:, iter + 1), options, iter, subsets, gaussK, Nx, Ny, Nz);
elseif options.weighted_mean && options.OSL_MLEM
    im_vectors.Weighted_MLEM(:, iter + 1) = deblur(im_vectors.Weighted_MLEM(:, iter + 1), options, iter, subsets, gaussK, Nx, Ny, Nz);
elseif options.TV && options.OSL_MLEM
    im_vectors.TV_MLEM(:, iter + 1) = deblur(im_vectors.TV_MLEM(:, iter + 1), options, iter, subsets, gaussK, Nx, Ny, Nz);
elseif options.AD && options.OSL_MLEM
    im_vectors.AD_MLEM(:, iter + 1) = deblur(im_vectors.AD_MLEM(:, iter + 1), options, iter, subsets, gaussK, Nx, Ny, Nz);
elseif options.APLS && options.OSL_MLEM
    im_vectors.APLS_MLEM(:, iter + 1) = deblur(im_vectors.APLS_MLEM(:, iter + 1), options, iter, subsets, gaussK, Nx, Ny, Nz);
elseif options.TGV && options.OSL_MLEM
    im_vectors.TGV_MLEM(:, iter + 1) = deblur(im_vectors.TGV_MLEM(:, iter + 1), options, iter, subsets, gaussK, Nx, Ny, Nz);
elseif options.NLM && options.OSL_MLEM
    im_vectors.NLM_MLEM(:, iter + 1) = deblur(im_vectors.NLM_MLEM(:, iter + 1), options, iter, subsets, gaussK, Nx, Ny, Nz);
end