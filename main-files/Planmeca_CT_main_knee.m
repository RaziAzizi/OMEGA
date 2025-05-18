%% MATLAB codes for CBCT reconstruction for the Planmeca data
% This example contains a simplified example for Planmeca CBCT data.
% Simplified refers to less number of selectable options. You can always
% add other options manually. This example uses PDHG with subsets and
% filtering. Other algorithms that have been left here are PKMA and PDDY.
% Regularization includes TV, NLM, RDP, GGMRF and the non-local variations.
% You can manually add any of the built-in priors if desired. You can
% simply run this file and select the desired measurement data and use the
% default parameters or modify the parameters accordingly and then run
% this.

clear
clear mex

% addpath(genpath('/pfs/lustrep2/users/vwettenh/OMEGA'))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% SCANNER PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%% Binning
% The level of binning used for the raw data. For example binning of 2
% reduces the size of the projections by two from both dimensions (e.g.
% 2048x3072 becomes 1024x1536).
options.binning = 1;

%%% Name of current datafile/examination
% This is used for (potential) naming purposes only
options.name = 'Planmeca_CT_data';

%%% Compute only the reconstructions
% If this file is run with this set to true, then the data load and
% sinogram formation steps are always skipped. Precomputation step is
% only performed if precompute_lor = true and precompute_all = true
% (below). Normalization coefficients are not computed even if selected.
options.only_reconstructions = false;

%%% Show status messages
% These are e.g. time elapsed on various functions and what steps have been
% completed. It is recommended to keep this 1.  Maximum value of 3 is
% supported.
options.verbose = 1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This should be the full path to the metadata-file of the examination you
% wish to reconstruct. Alternatively you can leave this empty and you will
% then be prompted for the metadata-file.
%options.fpath = '/research/users/razazizi/20230413160651Hip_120kV800mAsN500/metadata';
options.fpath = '/research/users/razazizi/20230414085533Lumbar_120kV400mAsN500/metadata';

if ~options.only_reconstructions || ~isfield(options,'SinM')
    options = loadPlanmecaData(options);
end

% NOTE: If you want to reduce the number of projections, you need to do
% this manually as outlined below:
lasku = 1;
options.SinM = options.SinM(:,:,1:lasku:options.nProjections);
options.angles = options.angles(1:lasku:options.nProjections);
options.pitchRoll = options.pitchRoll(1:lasku:options.nProjections,:);
options.x = options.x(1:lasku:options.nProjections,:);
options.y = options.y(1:lasku:options.nProjections,:);
options.z = options.z(1:lasku:options.nProjections,:);
options.nProjections = numel(options.angles);

% options.oOffsetX = 0;
% options.oOffsetY = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% IMAGE PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The image size is taken from the conf file, but you can manually adjust
% these if desired
%%% Reconstructed image pixel count (X/row-direction)
% options.Nx = 801;

%%% Y/column-direction
% options.Ny = 801;

%%% Z-direction (number of slices) (axial)
% options.Nz = 668;

% Use these two to rotate/flip the final image
%%% Flip the image (in horizontal direction)?
options.flip_image = true;

%%% How much is the image rotated (radians)?
% The angle (in radians) on how much the image is rotated BEFORE
% reconstruction, i.e. the rotation is performed in the detector space.
% Positive values perform the rotation in counter-clockwise direction
% options.offangle = (2.48*pi)/2;
options.offangle = (3*pi)/2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Use projection extrapolation
options.useExtrapolation = false;

%%% Use extended FOV
options.useEFOV = true;

% Use transaxial extended FOV (this is off by default)
options.transaxialEFOV = true;

% Use axial extended FOV (this is on by default. If both this and
% transaxialEFOV are false but useEFOV is true, the axial EFOV will be
% turned on)
options.axialEFOV = true;

% Same as above, but for extrapolation. Same default behavior exists.
options.transaxialExtrapolation = false;

% Same as above, but for extrapolation. Same default behavior exists.
options.axialExtrapolation = true;

% Setting this to true uses multi-resolution reconstruction when using
% extended FOV. Only applies to extended FOV!
options.useMultiResolutionVolumes = 2;

% This is the scale value for the multi-resolution volumes. The original
% voxel size is divided by this value and then used as the voxel size for
% the multi-resolution volumes. Default is 1/4 of the original voxel size.
options.multiResolutionScale = 1/4;

% options.eFOVLength = 0.2;

% options.extrapLength = 0.3;

% Performs the extrapolation and adjusts the image size accordingly
options = CTEFOVCorrection(options);

% Use offset-correction
options.offsetCorrection = false;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% RECONSTRUCTION PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% IMPLEMENTATIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Reconstruction implementation used
% 1 = Reconstructions in MATLAB (projector in a MEX-file), uses matrices.
% (Slow and memory intensive)
% 2 = Matrix-free reconstruction with OpenCL/ArrayFire (Recommended)
% (Requires ArrayFire. Compiles with MinGW ONLY when ArrayFire was compiled
% with MinGW as well (cannot use the prebuilt binaries)).
% 3 = Multi-GPU/device matrix-free OpenCL (OSEM & MLEM only).
% 4 = Matrix-free reconstruction with OpenMP (parallel), standard C++
% 5 = Matrix-free reconstruction with OpenCL (parallel)
% See the doc for more information:
% https://omega-doc.readthedocs.io/en/latest/implementation.html
options.implementation = 2;

% Applies to implementations 3 and 5 ONLY
%%% OpenCL platform used 
% NOTE: Use OpenCL_device_info() to determine the platform numbers and
% their respective devices with implementations 3 or 5.
options.platform = 0;

% Applies to implementations 2, 3 and 5 ONLY
%%% OpenCL/CUDA device used 
% NOTE: Use ArrayFire_OpenCL_device_info() to determine the device numbers
% with implementation 2.
% NOTE: Use OpenCL_device_info() to determine the platform numbers and
% their respective devices with implementations 3 or 5.
% NOTE: The device numbers might be different between implementation 2 and
% implementations 3 and 5
% NOTE: if you switch devices then you need to run the below line
% (uncommented) as well:
% clear mex
options.use_device = 0;

% Implementation 2 ONLY
%%% Use CUDA
% Selecting this to true will use CUDA kernels/code instead of OpenCL. This
% only works if the CUDA code was successfully built.
options.use_CUDA = false;

% Implementation 2 ONLY
%%% Use CPU
% Selecting this to true will use CPU-based code instead of OpenCL or CUDA.
options.use_CPU = false;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PROJECTOR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Type of projector to use for the geometric matrix
% 1 = Improved/accelerated Siddon's algorithm
% 2 = Orthogonal distance based ray tracer
% 3 = Volume of intersection based ray tracer
% 4 = Interpolation-based projector (ray- and voxel-based)
% 5 = Branchless distance-driven projector
% NOTE: You can mix and match most of the projectors. I.e. 41 will use
% interpolation-based projector for forward projection while improved
% Siddon is used for backprojection.
% See the doc for more information:
% https://omega-doc.readthedocs.io/en/latest/selectingprojector.html
options.projector_type = 4;

%%% Use mask
% The mask needs to be a binary mask (uint8 or logical) where 1 means that
% the pixel is included while 0 means it is skipped. Separate masks can be
% used for both forward and backward projection and either one or both can
% be utilized at the same time. E.g. if only backprojection mask is input,
% then only the voxels which have 1 in the mask are reconstructed.
% Currently the masks need to be a 2D image that is applied identically at
% each slice.
% Forward projection mask
% If nonempty, the mask will be applied. If empty, or completely omitted, no
% mask will be considered.
% options.maskFP = true(options.nRowsD,options.nColsD);
% Backprojection mask
% If nonempty, the mask will be applied. If empty, or completely omitted, no
% mask will be considered.
% Create a circle that fills the FOV:
% [columnsInImage, rowsInImage] = meshgrid(1:options.Nx, 1:options.Ny);
% centerX = options.Nx/2;
% centerY = options.Ny/2;
% radius = options.Nx/2 - 40;
% options.maskBP = uint8((rowsInImage - centerY).^2 ...
%     + (columnsInImage - centerX).^2 <= radius.^2);
% options.maskBP = repmat(options.maskBP, 1, 1, options.Nz);
% options.maskBP(:,:,1:5) = options.maskBP(:,:,1:5) * 0;

%%% Interpolation length (projector type = 4 only)
% This specifies the length after which the interpolation takes place. This
% value will be multiplied by the voxel size which means that 1 means that
% the interpolation length corresponds to a single voxel (transaxial)
% length. Larger values lead to faster computation but at the cost of
% accuracy. Recommended values are between [0.5 1].
options.dL = 1;


%%%%%%%%%%%%%%%%%%%%%%%%% RECONSTRUCTION SETTINGS %%%%%%%%%%%%%%%%%%%%%%%%%
%%% Number of iterations (all reconstruction methods)
options.Niter = 10;
%%% Save specific intermediate iterations
% You can specify the intermediate iterations you wish to save here. Note
% that this uses zero-based indexing, i.e. 0 is the first iteration (not
% the initial value). By default only the last iteration is saved. Only
% full iterations (epochs) can be saved.
options.saveNIter = [];
% Alternatively you can save ALL intermediate iterations by setting the
% below to true and uncommenting it. As above, only full iterations
% (epochs) are saved.
% options.save_iter = false;

%%% Number of subsets (all excluding MLEM and subset_type = 5)
options.subsets = 20;

%%% Subset type (n = subsets)
% 1 = Every nth (column) measurement is taken
% 2 = Every nth (row) measurement is taken (e.g. if subsets = 3, then
% first subset has measurements 1, 4, 7, etc., second 2, 5, 8, etc.) 
% 3 = Measurements are selected randomly
% 4 = (Sinogram only) Take every nth column in the sinogram
% 5 = (Sinogram only) Take every nth row in the sinogram
% 6 = Sort the LORs according to their angle with positive X-axis, combine
% n_angles together and have 180/n_angles subsets for 2D slices and
% 360/n_angles for 3D, see GitHub wiki for more information:
% https://github.com/villekf/OMEGA/wiki/Function-help#reconstruction-settings
% 7 = Form the subsets by using golden angle sampling
% 8 = Use every nth projection image
% 9 = Randomly select the projection images
% 10 = Use golden angle sampling to select the subsets (not recommended for
% PET)
% 11 = Use prime factor sampling to select the projection images
% Most of the time subset_type 8 is sufficient.
options.subset_type = 8;

%%% Initial value for the reconstruction
% load sedentexJaw100mAs_refRDP.mat
% options.x0 = pz;
options.x0 = ones(options.Nx, options.Ny, options.Nz) * 1e-4;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% RECONSTRUCTION ALGORITHMS %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reconstruction algorithms to use (choose only one algorithm and
% optionally one prior)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NON-REGULARIZED %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% LSQR
options.LSQR = false;

%%% Conjugate Gradient Least-squares (CGLS)
options.CGLS = false;

%%% Feldkamp-Davis-Kress (FDK)
options.FDK = false;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MAP-METHODS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% These algorithms can utilize any of the selected priors, though only one
% prior can be used at a time

%%% Preconditioner Krasnoselskii-Mann algorithm (PKMA)
% Supported by implementations 1, 2, 4, and 5
options.PKMA = false;

%%% Primal-dual hybrid gradient (PDHG)
% Supported by implementations 1, 2, 4, and 5
options.PDHG = true;

%%% Primal-dual hybrid gradient (PDHG) with L1 minimization
% Supported by implementations 1, 2, 4, and 5
options.PDHGL1 = false;

%%% Primal-dual Davis-Yin (PDDY)
% Supported by implementation 2
options.PDDY = false;

options.FISTA = false;

options.SART = false;

options.SAGA = false;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PRIORS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Total Generalized Variation (TGV) prior
options.TGV = false;

%%% Proximal TV
options.ProxTV = false;

%%% Non-local Means (NLM) prior
options.NLM = false;

%%% Relative difference prior
options.RDP = false;

%%% Generalized Gaussian Markov random field (GGMRF) prior
options.GGMRF = false;


%%%%%%%%%%%%%%%%%%%%%%%%%%%% ENFORCE POSITIVITY %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Applies to PDHG, PDHGL1, PDDY, FISTA, FISTAL1, MBSREM, MRAMLA, PKMA
% Enforces positivity in the estimate after each iteration
options.enforcePositivity = true;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PDHG PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Primal value
% If left zero, or empty, it will be automatically computed
options.tauCP = 0;%1/387509.406250;%2.56643271226464e-06;1/(57757.222656);%1/2;%
% options.tauCP = [1/202842;1/8771076;1/9235887];
% options.tauCP = [1/202842;1/8771076;1/9235887];
% Primal value for filtered iterations, applicable only if
% options.precondTypeMeas[2] = true. As with above, automatically computed
% if left zero or empty.
options.tauCPFilt = 0;
% options.tauCPFilt = [1/4298;1/57055;1/67568];
% options.tauCPFilt = 1/4383.810059;
% options.tauCPFilt = 1/812.362183;
% options.tauCPFilt = 1/147.669769;
% Dual value. Recommended to set at 1.
options.sigmaCP = 1;
% Next estimate update variable
options.thetaCP = 1;

options.sigma2CP = 3;

% Use adaptive update of the primal and dual variables
% Currently only one method available
% Setting this to 1 uses an adaptive update for both the primal and dual
% variables.
% Can lead to unstable behavior when using with multi-resolution!
% Minimal to none benefit with filtering-based preconditioner
options.PDAdaptiveType = 0;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PRECONDITIONERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Applies to PDHG, PDHGL1, PDHGKL, PKMA, MBSREM, MRAMLA, PDDY, FISTA and
%%% FISTAL1
% Measurement-based preconditioners
% precondTypeMeas(1) = Diagonal normalization preconditioner (1 / (A1))
% precondTypeMeas(2) = Filtering-based preconditioner
options.precondTypeMeas = [false;true];

% Number of filtering iterations
% Applies to both precondTypeMeas(2) and precondTypeImage(6)
options.filteringIterations = 100;


options.precondTypeImage = [false;false;false;false;false;false;false];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PKMA PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Relaxation parameter for PKMA
% If a scalar (or an empty) value is used, then the relaxation parameter is
% computed automatically as lambda(i) = (1 / ((i - 1)/20 + 1)) / 10000,
% where i is the iteration number. The input number thus has no effect.
% If, on the other hand, a vector is input then the input lambda values are
% used as is without any modifications (the length has to be at least the
% number of iterations).
options.lambda = 0;
% options.rho_PKMA = .95;%0.55%3%.45/PKMA%1.4/MBSREM
% options.delta_PKMA = 20;%1.5
% options.delta2_PKMA = 100;%100/PKMA%1.5/MBSREM
% options.lambda = zeros(options.Niter,1);
% for i = 0 : options.Niter - 1
%     if sum(options.precondTypeMeas) == 0
%         if (options.precondTypeImage(2) || options.precondTypeImage(3)) && ~options.precondTypeImage(7)
%             options.lambda(i + 1) = 1 / ((1/35) * i + 1) / 1 * 5.6e-5;%1 / ((i)/12 + 1);%exp(1 / (i/35 + 1)-1);%
%         elseif (options.precondTypeImage(2) || options.precondTypeImage(3)) && options.precondTypeImage(7)
%             options.lambda(i + 1) = 1 / ((1/35) * i + 1) / 1 * 5e2;
%         elseif options.precondTypeImage(7)
%             options.lambda(i + 1) = 1 / ((1/35) * i + 1) / 1 * 2.5e-0;
%         elseif options.precondTypeImage(1)
%             options.lambda(i + 1) = 1 / ((1/35) * i + 1) / 1 * 2.5e-6;
%         % elseif sum(options.precondTypeImage) == 0 && ~options.ROSEM
%         %     options.lambda(i + 1) = 1 / ((1/35) * i + 1) / 1 * 5.5e-7;
%         else
%             options.lambda(i + 1) = 1 / ((1/35) * i + 1) / 1;
%         end
%     else
%         if options.precondTypeMeas(1)
%             if (options.precondTypeImage(2) || options.precondTypeImage(3)) && ~options.precondTypeImage(7)
%                 options.lambda(i + 1) = 1 / ((1/35) * i + 1) / 1 * 2e-2;
%             elseif (options.precondTypeImage(2) || options.precondTypeImage(3)) && options.precondTypeImage(7)
%                 options.lambda(i + 1) = 1 / ((1/35) * i + 1) / 1 * 9e4;
%             end
%         else
%             if (options.precondTypeImage(2) || options.precondTypeImage(3)) && ~options.precondTypeImage(7)
%                 options.lambda(i + 1) = 1 / ((1/35) * i + 1) / 1 * 1.1e-4;
%             elseif (options.precondTypeImage(2) || options.precondTypeImage(3)) && options.precondTypeImage(7)
%                 options.lambda(i + 1) = 1 / ((1/35) * i + 1) / 1 * 5e2;
%             end
%         end
%     end
% end

%%% Step size (alpha) parameter for PKMA
% If a scalar (or an empty) value is used, then the alpha parameter is
% computed automatically as alpha_PKMA(oo) = 1 + (options.rho_PKMA *((i -
% 1) * options.subsets + ll)) / ((i - 1) * options.subsets + ll +
% options.delta_PKMA), where i is the iteration number and l the subset
% number. The input number thus has no effect. options.rho_PKMA and
% options.delta_PKMA are defined below.
% If, on the other hand, a vector is input then the input alpha values are
% used as is without any modifications (the length has to be at least the
% number of iterations * number of subsets).
options.alpha_PKMA = 0;

%%% rho_PKMA
% This value is ignored if a vector input is used with alpha_PKMA
options.rho_PKMA = 0.95;

%%% delta_PKMA
% This value is ignored if a vector input is used with alpha_PKMA
options.delta_PKMA = 100;


%%%%%%%%%%%%%%%%%%%%%%%%% REGULARIZATION PARAMETER %%%%%%%%%%%%%%%%%%%%%%%%
%%% The regularization parameter for ALL regularization methods (priors)
% 50-100 is good starting point for NLM
% ~1 is good starting region for RDP and NLRD
% options.beta = 20; % NLRDP
options.beta = 0.13; % TV
% options.beta = 250; % NLM
% options.beta = 1; % NLTV


%%%%%%%%%%%%%%%%%%%%%%%%% NEIGHBORHOOD PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%
%%% How many neighboring pixels are considered
% With MRP, QP, L, FMH, NLM and weighted mean
% E.g. if Ndx = 1, Ndy = 1, Ndz = 0, then you have 3x3 square area where
% the pixels are taken into account (I.e. (Ndx*2+1)x(Ndy*2+1)x(Ndz*2+1)
% area).
% NOTE: Currently Ndx and Ndy must be identical.
% For NLM this is often called the "search window".
options.Ndx = 5;
options.Ndy = 5;
options.Ndz = 0;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TGV PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% TGV weights
% First part
options.alpha0TGV = 0.15;
% Second part (symmetrized derivative)
options.alpha1TGV = 0.02;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NLM PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Filter parameter
% Higher values smooth the image, smaller values make it sharper
% options.sigma = 1.5;
options.sigma = 3e-3;

%%% Patch radius
options.Nlx = 1;
options.Nly = 1;
options.Nlz = 0;

%%% Standard deviation of the Gaussian filter
options.NLM_gauss = 1;

%%% Adaptive NL methods
options.NLAdaptive = false;

%%% Summed constant for adaptive NL
options.NLAdaptiveConstant = 5.0e-6;

% By default, the original NLM is used. You can, however, use another
% potential function by selecting ONE of the options below.
%%% Use Non-local total variation (NLTV)
% If selected, will overwrite regular NLM regularization as well as the
% below MRP version.
options.NLTV = true;

%%% Use Non-local relative difference (NLRD)
options.NLRD = false;

%%% Use Non-local Lange prior (NLLange)
options.NLLange = false;

% Tuning parameter for Lange
options.SATVPhi = 10;

%%% Use Non-local GGMRF (NLGGMRF)
options.NLGGMRF = false;

%%% Use MRP algorithm (without normalization)
% I.e. gradient = im - NLM_filtered(im)
options.NLM_MRP = false;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% RDP PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Edge weighting factor
% Note that this affects NLRD as well
% Higher values sharpen the image, lower values smooth it
options.RDP_gamma = 15;

options.RDPIncludeCorners = false;

options.RDP_use_anatomical = false;

options.RDP_reference_image = 'sedentexJaw100mAs_refRDP.mat';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GGMRF PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% GGMRF parameters
% See the original article for details
% These affect the NLGGMRF as well
options.GGMRF_p = 2;
options.GGMRF_q = 1.15;
options.GGMRF_c = .0005;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Store the intermediate forward projections. Unlike image estimates, this
% also stores subiteration results.
options.storeFP = false;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% OPENCL DEVICE INFO %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Implementation 2
% Uncomment the below line and run it to determine the available device
% numbers
% ArrayFire_OpenCL_device_info()

%%% Implementation 3
% Uncomment the below line and run it to determine the available platforms,
% their respective numbers and device numbers
% OpenCL_device_info()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load sedentexJaw100kV100mAs_PMFDK_HUcorrection.mat
% 
% options.x0 = x0(:);

% load knee_phantom_fdk_nocutting
% options.x0 = single(fdk(:) + 1000) / 55000;

options.storeResidual = false;

% options.loadTOF = false;

options.useFDKWeights = false;

options.FISTA_acceleration = false;

options.use2DTGV = false;

options.useL2Ball = true;

options.powerIterations = 10;

options.FISTAType = 2;

options.stochasticSubsetSelection = false;


% Available windows are: hamming, hann, blackman, nuttal, parzen, cosine, gaussian, and shepp-logan
options.filterWindow = 'hamming';

%%

t = tic;
% pz is the reconstructed image volume
% recPar is a short list of various reconstruction parameters used,
% useful when saving the reconstructed data with metadata
% classObject is the used class object in the reconstructions,
% essentially modified version of the input options-struct
% fp are the forward projections, if stored
% the primal-dual gap can be also be stored and is the variable after
% fp
[pz, recPar, ~, fp, res] = reconstructions_mainCT(options);
t = toc(t)

% Convert to HU-values
z = int16(pz{1}(:,:,:,end) * 55000) - 1000;

volume3Dviewer(z, [-1000 2000], [0 0 1])

% save('/pfs/lustref1/flash/project_462000633/testi.mat', '-v7.3', 'z', 't','recPar')
% 
% exit