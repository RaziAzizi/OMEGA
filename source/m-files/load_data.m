function [varargout] = load_data(options)
%% Load PET data
% Loads either ASCII, ROOT, or LMF GATE format data or the Inveon list-mode
% (.lst) data into MATLAB/Octave.
%
% Stores the sinogram data, unless raw data has been selected (i.e.
% options.use_raw_data = true). Raw data is stored only if either
% options.use_raw_data = true or options.store_raw_data = true. Outputs raw
% data if it has been loaded, otherwise sinogram data, i.e. sinogram data
% can be output from this function by setting options.store_raw_data =
% false. Loads also the delayed coincidences, true randoms, true scatter
% and true coincidences separately if they are selected.
%
% The input struct options should include all the relevant information
% (machine properties, path, names, what are saved, etc.).
%
% The output data is saved in a mat-file in the current working directory.
%
% EXAMPLES:
%   coincidences = load_data(options);
%   [coincidences, delayed_coincidences, true_coincidences,
%   scattered_coincidences, random_coincidences] = load_data(options);
%   [coincidences, ~, ~, ~, ~, x, y, z] = load_data(options);
%
% OUTPUT:
%   coincidences = A cell matrix containing the raw list-mode data for each
%   time step. The contents of the cell matrix are a vector containing the
%   coincidences of each unique LOR (no duplicates, e.g. coincidences
%   from detector 1 to 2 and 2 to 1 are summed up)
%   delayed_coincidences = Same as above, but for delayed coincidences
%   true_coincidences = True coincidences (GATE only)
%   scattered_coincidences = Scattered coincidences (GATE only)
%   random_coincidences = True random coincidences (GATE only)
%   x = The x coordinates of the hits, i.e. the location of the detected
%   coincidence in the x-axis. The coordinates are output only if the
%   function has 8 outputs (see last example above). Only ASCII and ROOT
%   data are supported (GATE only).
%   y = Same as above, but for y-coordinates
%   z = Axial coordinate
%
% See also form_sinograms, initial_michelogram

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2020 Ville-Veikko Wettenhovi
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program. If not, see <https://www.gnu.org/licenses/>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargout > 9
    error('Too many output arguments, there can be at most nine!')
end

if ~isfield(options,'TOF_bins') || options.TOF_bins == 0
    options.TOF_bins = 1;
end
if ~isfield(options,'TOF_width') || options.TOF_bins == 0
    options.TOF_width = 0;
end
if ~isfield(options,'legacyROOT')
    options.legacyROOT = false;
end
if ~isfield(options,'axial_multip')
    options.axial_multip = 1;
end
if ~isfield(options,'nLayers')
    options.nLayers = 1;
end
options = setMissingValues(options);


if mod(options.TOF_bins, 2) == 0 && options.TOF_bins > 1
    error('Number of TOF bins has to be odd')
end

if numel(options.partitions) > 1
    partitions = numel(options.partitions);
elseif isempty(options.partitions)
    partitions = 1;
else
    partitions = options.partitions;
end
alku = double(options.start);
loppu = options.end;
if isinf(loppu)
    loppu = 1e9;
end
vali = double((loppu - alku)/partitions);
% tot_time = options.tot_time;
% if options.useIndexBasedReconstruction && ~options.use_root
%     error('Detector indices can only be saved from ROOT data!')
% end
if options.useIndexBasedReconstruction
    if options.obtain_trues
        warning('Trues cannot be obtained with index-based reconstruction on!')
    end
    if options.store_randoms
        warning('Randoms cannot be stored with index-based reconstruction on!')
    end
    if options.store_scatter
        warning('Scatter cannot be stored with index-based reconstruction on!')
    end
end
if options.useIndexBasedReconstruction && nargout < 7
    error('Too little of output arguments when extracting detector indices! There must be at least 7 outputs!')
end

if isscalar(partitions) && partitions > 1
    if isinf(options.end)
        error('End time is infinity, but more than one time-step selected. Use either one time-step or input a finite end time.')
    end
    options.partitions = repmat(vali, options.partitions, 1);
end

Nt = numel(partitions);
% partitions = options.partitions;

if ispc && options.use_root
    options.legacyROOT = true;
end

[~,~,~,dx,dy,dz,bx,by,bz] = computePixelSize([options.FOVa_x; options.FOVa_y; options.axial_fov], [options.Nx; options.Ny; options.Nz], ...
    [options.oOffsetX; options.oOffsetY; options.oOffsetZ], 'double');

if numel(dx) > 1
    dx = dx(0);
    dy = dy(0);
    dz = dz(0);
    bx = bx(0);
    by = by(0);
    bz = bz(0);
end

% machine_name = options.machine_name;
name = options.name;
detectors = options.detectors;
% trues = [];
% scatter = [];
% randoms = [];
% delays = [];

trues_index = logical([]);
scatter_index = logical([]);
randoms_index = logical([]);
SC = [];
RA = [];
totSinos = options.TotSinos;
if options.span == 1
    totSinos = options.rings(end).^2;
end
sinoSize = uint64(options.Ndist * options.Nang(end) * totSinos);

temp = options.pseudot;
if ~isempty(temp) && temp > 0
    pseudot = zeros(sum(options.pseudot),1,'uint16');
    for kk = uint16(1) : temp
        pseudot(kk) = uint16(options.cryst_per_block + 1) * kk - 1;
    end
elseif temp == 0
    pseudot = [];
end
raw_SinM = zeros(options.Ndist, options.Nang(end), totSinos, options.nLayers^2, options.TOF_bins, Nt, 'uint16');
% raw_SinM = cell(options.partitions,1);
% ix = cellfun('isempty',raw_SinM);
% raw_SinM(ix) = {zeros(options.Ndist, options.Nang, totSinos, options.TOF_bins, 'uint16')};
if options.obtain_trues
    SinTrues = zeros(options.Ndist, options.Nang, totSinos, options.nLayers^2, options.TOF_bins, Nt, 'uint16');
else
    SinTrues = uint16(0);
end
if options.store_scatter
    SinScatter = zeros(options.Ndist, options.Nang, totSinos, options.nLayers^2, options.TOF_bins, Nt, 'uint16');
else
    SinScatter = uint16(0);
end
if options.store_randoms
    SinRandoms = zeros(options.Ndist, options.Nang, totSinos, options.nLayers^2, options.TOF_bins, Nt, 'uint16');
else
    SinRandoms = uint16(0);
end
if options.randoms_correction
    SinD = zeros(options.Ndist, options.Nang, totSinos, options.nLayers^2, Nt, 'uint16');
else
    SinD = uint16(0);
end
if ~isfield(options,'store_raw_data')
    options.store_raw_data = false;
end

TOF = options.TOF_bins > 1;

if nargout >= 6 && ~options.useIndexBasedReconstruction
    store_coordinates = true;
else
    store_coordinates = false;
end
max_time = 0;


disp('Beginning data load')

%% Load Inveon list-mode data
if options.use_machine == 1

    % Select the list-mode file
    if exist(options.fpath,'file') ~= 2
        [options.file, options.fpath] = uigetfile('*.lst','Select Inveon list-mode datafile');
        if isequal(options.file, 0)
            error('No file was selected')
        end
        nimi = [options.fpath options.file];
    else
        nimi = [options.fpath];
    end

    if options.verbose
        tic
    end

    if store_coordinates
        options.store_raw_data = true;
    end

    % if partitions > 1
    %     save_string_raw = [machine_name '_measurements_' name '_' num2str(partitions) 'timepoints_for_total_of_' num2str(loppu - alku) 's_raw_listmode.mat'];
    % else
    %     save_string_raw = [machine_name '_measurements_' name '_static_raw_listmode.mat'];
    % end
    %
    % variableList = {'delayed_coincidences'};


    f = dir(nimi);
    pituus = uint64(f.bytes/6);

    if options.verbose
        disp('Data load started')
    end
    if store_coordinates || options.useIndexBasedReconstruction
        storeL = true;
    else
        storeL = false;
    end
    % tic
    % Load the data from the binary file
    [LL1, LL2, tpoints, DD1, DD2, raw_SinM, SinD] = inveon_list2matlab(nimi,(partitions),(alku),(loppu),pituus, uint32(detectors), options.randoms_correction, sinoSize, ...
        options.store_raw_data, uint32(options.Ndist), uint32(options.Nang), uint32(options.ring_difference), uint32(options.span), uint32(cumsum(options.segment_table)), ...
        Nt, int32(options.ndist_side), storeL);
    % toc

    % If no time steps, the output matirx LL1 is a options.detectors x
    % options.detectors matrix in unsigned short (u16) format.
    % Contains counts for each detector pair
    % if options.store_raw_data
    if store_coordinates
        if Nt > 1
            coordinate = cell(Nt,1);
        end
        LL1(LL1==0) = [];
        LL2(LL2==0) = [];
        ring_number1 = uint16(idivide(LL1-1, options.det_per_ring)) + 1;
        ring_number2 = uint16(idivide(LL2-1, options.det_per_ring)) + 1;
        ring_pos1 = uint16(mod(LL1-1, options.det_per_ring)) + 1;
        ring_pos2 = uint16(mod(LL2-1, options.det_per_ring)) + 1;
        clear LL1 LL2
        [x, y] = detector_coordinates(options);
        z_length = single(options.rings * options.cr_pz);
        z = linspace(single(0), z_length, options.rings + 1)';
        z = z(2:end) - z(2) / 2;
        z = z - options.axial_fov/2 + (options.axial_fov - (options.cr_pz*options.rings))/2;
        % if min(z(:)) == 0
        %     z = z + (options.axial_fov - (options.rings) * options.cr_pz)/2 + options.cr_pz/2;
        % end
        x = single(x);
        y = single(y);
        if Nt > 1
            tpoints(tpoints==0) = [];
            for uu = 1 : Nt
                ind1 = find(tpoints == uu, 1, 'first');
                ind2 = find(tpoints == uu, 1, 'last');
                tempPos1 = ring_pos1(ind1:ind2);
                tempPos2 = ring_pos2(ind1:ind2);
                tempNum1 = ring_number1(ind1:ind2);
                tempNum2 = ring_number2(ind1:ind2);
                coordinate{uu + 1} = [x(tempPos1)'; y(tempPos1)'; z(tempNum1)'; x(tempPos2)'; y(tempPos2)'; z(tempNum2)'];
            end
        else
            coordinate = [x(ring_pos1)'; y(ring_pos1)'; z(ring_number1)'; x(ring_pos2)'; y(ring_pos2)'; z(ring_number2)'];
        end
        % y_coordinate = [y(ring_pos1) y(ring_pos2)];
        clear ring_pos1 ring_pos2 ring_number1 ring_number2
        % z_coordinate = [z(ring_number1) z(ring_number2)];
        % options.store_raw_data = false;
        if options.randoms_correction
            if Nt > 1
                Rcoordinate = cell(Nt,1);
            end
            DD1(DD1==0) = [];
            DD2(DD2==0) = [];
            ring_number1 = uint16(idivide(DD1-1, options.det_per_ring)) + 1;
            ring_number2 = uint16(idivide(DD2-1, options.det_per_ring)) + 1;
            ring_pos1 = uint16(mod(DD1-1, options.det_per_ring)) + 1;
            ring_pos2 = uint16(mod(DD2-1, options.det_per_ring)) + 1;
            clear DD1 DD2
            if Nt > 1
                for uu = 1 : Nt
                    ind1 = find(tpoints == uu, 1, 'first');
                    ind2 = find(tpoints == uu, 1, 'last');
                    tempPos1 = ring_pos1(ind1:ind2);
                    tempPos2 = ring_pos2(ind1:ind2);
                    tempNum1 = ring_number1(ind1:ind2);
                    tempNum2 = ring_number2(ind1:ind2);
                    Rcoordinate{uu + 1} = [x(tempPos1)'; y(tempPos1)'; z(tempNum1)'; x(tempPos2)'; y(tempPos2)'; z(tempNum2)'];
                end
            else
                Rcoordinate = [x(ring_pos1)'; y(ring_pos1)'; z(ring_number1)'; x(ring_pos2)'; y(ring_pos2)'; z(ring_number2)'];
            end
            % y_Rcoordinate = [y(ring_pos1) y(ring_pos2)];
            clear ring_pos1 ring_pos2 ring_number1 ring_number2
            % z_Rcoordinate = [z(ring_number1) z(ring_number2)];
            % clear ring_number1 ring_number2
        end
        %     else
        %         if partitions == 1
        %             prompts = LL1';
        %             if options.randoms_correction
        %                 delays = DD1';
        %             end
        %             clear LL1 LL2 DD1 DD2
        %             % Else the outputs are vectors LL1 and LL2 containing the detector
        %             % pairs for all the counts.
        %             % Form a similar (sparse) detectors x detectors matrix
        %         else
        %             prompts = cell(partitions,1);
        %             if options.randoms_correction
        %                 delays = cell(partitions,1);
        %             end
        %             %         ll = 1;
        %             % Obtain the counts that were within a specified time-window
        %             for kk = 1 : partitions
        %                 apu1 = LL1(tpoints(kk) + 1:tpoints(kk+1));
        %                 apu1(apu1 == 0) = [];
        %                 apu2 = LL2(tpoints(kk) + 1:tpoints(kk+1));
        %                 apu2(apu2 == 0) = [];
        %                 prompts{kk} = accumarray([apu1 apu2],1,[options.detectors options.detectors],[],[],true);
        %                 if options.randoms_correction
        %                     apu1 = DD1(tpoints(kk) + 1:tpoints(kk+1));
        %                     apu1(apu1 == 0) = [];
        %                     apu2 = DD2(tpoints(kk) + 1:tpoints(kk+1));
        %                     apu2(apu2 == 0) = [];
        %                     delays{kk} = accumarray([apu1 apu2],1,[options.detectors options.detectors],[],[],true);
        %                 end
        %                 %             ll = ll + tpoints(kk+1);
        %             end
        %             clear LL1 LL2 DD1 DD2 apu1 apu2
        %         end
    elseif options.useIndexBasedReconstruction
        LL1(LL1==0) = [];
        LL2(LL2==0) = [];
        axIndices = [uint16(idivide(LL1'-1, options.det_per_ring));uint16(idivide(LL2'-1, options.det_per_ring))];
        trIndices = [uint16(mod(LL1'-1, options.det_per_ring));uint16(mod(LL2'-1, options.det_per_ring))];
        if options.randoms_correction
            DD1(DD1==0) = [];
            DD2(DD2==0) = [];
            DaxIndices = [uint16(idivide(DD1'-1, options.det_per_ring));uint16(idivide(DD2'-1, options.det_per_ring))];
            DtrIndices = [uint16(mod(DD1'-1, options.det_per_ring));uint16(mod(DD2'-1, options.det_per_ring))];
        end
    end
    % end

    %% GATE data
elseif options.use_machine == 0
    if options.use_ASCII && options.use_LMF && options.use_root
        warning('ASCII, LMF and ROOT selected, using only ASCII')
        options.use_LMF = false;
        options.use_root = false;
    elseif options.use_ASCII && options.use_LMF
        warning('Both ASCII and LMF selected, using only ASCII')
        options.use_LMF = false;
    elseif options.use_ASCII && options.use_root
        warning('Both ASCII and ROOT selected, using only ASCII')
        options.use_root = false;
    elseif options.use_LMF && options.use_root
        warning('Both LMF and ROOT selected, using only LMF')
        options.use_root = false;
    elseif options.use_ASCII == false && options.use_LMF == false && options.use_root == false
        error('Error: Neither ASCII, LMF nor ROOT data selected')
    end
    if options.verbose
        tic
    end

    if isunix
        if length(options.fpath) > 1 && ~strcmp('/',options.fpath(end))
            options.fpath = [options.fpath '/'];
        end
    elseif ispc
        if length(options.fpath) > 1 && ~strcmp('\',options.fpath(end)) && ~strcmp('/',options.fpath(end))
            options.fpath = [options.fpath '\'];
        end
    else
        if length(options.fpath) > 1 && ~strcmp('/',options.fpath(end))
            options.fpath = [options.fpath '/'];
        end
    end

    if TOF
        FWHM = double(options.TOF_FWHM / (2 * sqrt(2 * log(2))));
    else
        FWHM = 0;
    end

    fpath = options.fpath;
    blocks_per_ring = options.blocks_per_ring;
    cryst_per_block = options.cryst_per_block;
    if ~isfield(options,'cryst_per_block_axial')
        options.cryst_per_block_axial = options.cryst_per_block;
    end
    cryst_per_block_z = options.cryst_per_block_axial;
    machine_name = options.machine_name;
    rings = options.rings;
    det_per_ring = options.det_per_ring;
    source = options.source;
    det_per_ring = uint32(det_per_ring);
    if ~isfield(options,'transaxial_multip')
        options.transaxial_multip = 1;
    end
    transaxial_multip = options.transaxial_multip;
    % Total number of detectors
    detectors = uint32(options.det_per_ring.*(rings - sum(options.pseudot)));

    % if (options.store_raw_data)
    %     if options.partitions > 1
    %         prompts = cell(partitions,1);
    %         if options.obtain_trues
    %             trues = cell(partitions,1);
    %         end
    %         if options.store_scatter
    %             scatter = cell(partitions,1);
    %         end
    %         if options.store_randoms
    %             randoms = cell(partitions,1);
    %         end
    %         if options.randoms_correction && ~options.use_LMF
    %             delays = cell(partitions,1);
    %         end
    %     else
    %         prompts = zeros(detectors, detectors, 'uint16');
    %         if options.obtain_trues
    %             trues = zeros(detectors, detectors, 'uint16');
    %         end
    %         if options.store_scatter
    %             scatter = zeros(detectors, detectors, 'uint16');
    %         end
    %         if options.store_randoms
    %             randoms = zeros(detectors, detectors, 'uint16');
    %         end
    %         if options.randoms_correction && ~options.use_LMF
    %             delays = zeros(detectors, detectors, 'uint16');
    %         end
    %     end
    % end
    if source
        % if options.obtain_trues
        %     C = cell(7,partitions);
        % else
        %     C = cell(6,partitions);
        % end
        % C(:) = {zeros(options.Nx, options.Ny, options.Nz, 'uint16')};
        C = zeros(options.Nx, options.Ny, options.Nz, Nt, 'uint16');
        if options.store_scatter && ~options.use_LMF
            SC = zeros(options.Nx, options.Ny, options.Nz, Nt, 'uint16');
            % else
            %     SC = zeros(1, 1, 1, 1, 'uint16');
        end
        if options.store_randoms && ~options.use_LMF
            RA = zeros(options.Nx, options.Ny, options.Nz, Nt, 'uint16');
            % else
            %     RA = zeros(1, 1, 1, 1, 'uint16');
        end
    end

    % variableList = {'true_coincidences','scattered_coincidences','random_coincidences','delayed_coincidences'};

    % if numel(options.partitions) == 1
    %     partitions = linspace(alku, loppu, options.partitions + 1);
    % else
    %     partitions = options.partitions;
    % end

    if Nt > 1
        if store_coordinates
            coordinate = cell(Nt,1);
            % y_coordinate = cell(partitions,1);
            % z_coordinate = cell(partitions,1);
            if options.randoms_correction
                Rcoordinate = cell(Nt,1);
            end
        end
    else
        % lisays = uint16(1);
        if store_coordinates
            coordinate = [];
            % y_coordinate = [];
            % z_coordinate = [];
            if options.randoms_correction
                Rcoordinate = [];
            end
        end
    end
    if options.useIndexBasedReconstruction
        trIndices = [];
        axIndices = [];
        if options.randoms_correction
            trIndices = [];
            DaxIndices = [];
        end
    end



    %% LMF data
    if options.use_LMF
        warning('LMF support has been deprecated! Functionality is not guaranteed!')
        blocks_per_ring = uint32(blocks_per_ring);
        cryst_per_block = uint32(cryst_per_block);
        R_bits = int32(options.R_bits);
        M_bits = int32(options.M_bits);
        S_bits = int32(options.S_bits);
        C_bits = int32(options.C_bits);
        L_bits = int32(options.L_bits);
        coincidence_window = options.coincidence_window;
        header_bytes = int32(options.header_bytes);
        data_bytes = int32(options.data_bytes);

        coincidence_window = coincidence_window / options.clock_time_step;

        % if partitions > 1
        %     save_string_raw = [machine_name '_measurements_' name '_' num2str(partitions) 'timepoints_for_total_of_' num2str(loppu - alku) 's_raw_LMF.mat'];
        % else
        %     save_string_raw = [machine_name '_measurements_' name '_static_raw_LMF.mat'];
        % end

        % Take coincidence data
        [~,~,ext] = fileparts(fpath);
        if strcmp(ext,'ccs')
            fnames = dir(fpath);
        else
            fnames = dir([fpath '/*.ccs']);
        end
        if size(fnames,1) == 0
            fpath = pwd;
            fpath = [fpath '/'];
            fnames = dir([fpath '*.ccs']);
            if size(fnames,1) == 0
                error('No LMF (.ccs) files were found. Check your filepath (options.fpath) or current folder.')
            end
        end

        % Bit locations
        bina = char(zeros(1,R_bits));
        for kk = 1 : R_bits
            bina(kk) = '1';
        end
        R_length = int32(bin2dec(bina));
        bina = char(zeros(1,M_bits));
        for kk = 1 : M_bits
            bina(kk) = '1';
        end
        M_length = int32(bin2dec(bina));
        bina = char(zeros(1,S_bits));
        for kk = 1 : S_bits
            bina(kk) = '1';
        end
        S_length = int32(bin2dec(bina));
        bina = char(zeros(1,C_bits));
        for kk = 1 : C_bits
            bina(kk) = '1';
        end
        C_length = int32(bin2dec(bina));

        if options.verbose
            disp('Data load started')
        end

        % Go through all the files
        for lk=1:length(fnames)

            pituus = int64((fnames(lk).bytes - header_bytes) / data_bytes);

            nimi = [fpath fnames(lk).name];

            [L1, L2, tpoints, A, int_loc, Ltrues, Lscatter, Lrandoms, trues_index, raw_SinM, SinTrues, SinScatter, SinRandoms] = gate_lmf_matlab(nimi,Nt,alku,loppu,pituus, uint32(detectors), blocks_per_ring, ...
                cryst_per_block, det_per_ring, uint32(options.linear_multip), header_bytes, data_bytes, R_bits, M_bits, S_bits, C_bits, L_bits, R_length, M_length, ...
                S_length, C_length, source, coincidence_window, options.clock_time_step, partitions, options.obtain_trues, options.store_scatter, options.store_randoms, ...
                uint32(cryst_per_block_z), uint32(transaxial_multip), uint32(options.rings), sinoSize, ...
                uint32(options.Ndist), uint32(options.Nang), uint32(options.ring_difference), uint32(options.span), uint32(cumsum(options.segment_table)), ...
                uint64(Nt), sinoSize * uint64(options.TOF_bins), int32(options.ndist_side), options.store_raw_data, raw_SinM, SinTrues, SinScatter, ...
                SinRandoms, int32(options.det_w_pseudo), int32(sum(options.pseudot)));

            trues_index = logical(trues_index);
            scatter_index = false(size(A,1));
            randoms_index = false(size(A,1));

            % An ideal image can be formed with this, see the file
            % visualize_pet.m
            ll = 1;
            if source && int_loc(1) > 0

                for jj = int_loc(1) : min(int_loc(2),partitions)
                    if partitions > 1
                        S = A(tpoints(ll) + 1:tpoints(ll+1),:);
                        inde = any(S,2);
                        if options.obtain_trues
                            trues_index = logical(Ltrues(tpoints(ll) + 1:tpoints(ll+1),:));
                            trues_index = trues_index(inde);
                        end
                        S = single(S(inde,:));
                    else
                        inde = any(A,2);
                        if options.obtain_trues
                            trues_index = logical(trues_index(inde));
                        end
                        A = A(inde,:);
                        S = single(A);
                    end
                    [C, SC, RA] = formSourceImage(options, S, jj, trues_index, scatter_index, randoms_index, C, SC, RA, tpoints(ll) + 1, tpoints(ll+1));
                    ll = ll + 1;
                end
            end

            % if (options.store_raw_data)
            %     if partitions == 1
            %         prompts = prompts + L1;
            %         if options.obtain_trues
            %             trues = trues + Ltrues;
            %         end
            %         if options.store_scatter
            %             scatter = scatter + Lscatter;
            %         end
            %         if options.store_randoms
            %             randoms = randoms + Lrandoms;
            %         end
            %     else
            %         % forms a sparse matrix containing all the events at different LORs
            %         % e.g. a value at [325 25100] contains the number of events detected at
            %         % detectors 325 (detector 1) and 25100 (detector 2)
            %         % a value at [25100 325] on the other hand tells the number of events when
            %         % 25100 is detector 1 and 325 detector 2
            %         if  int_loc(1) > 0
            %             ll = 1;
            %             for kk = int_loc(1) : min(int_loc(2),partitions)
            %                 %                 index = find(M(:,time_index)<tpoints(kk),1,'last');
            %                 apu1 = L1(tpoints(ll) + 1:tpoints(ll+1));
            %                 apu2 = L2(tpoints(ll) + 1:tpoints(ll+1));
            %                 inde = any(apu1,2);
            %                 apu1 = apu1(inde,:);
            %                 apu2 = apu2(inde,:);
            %                 if isempty(prompts{kk})
            %                     prompts{kk} = accumarray([apu1 apu2],1,[detectors detectors],[],[],true);
            %                 else
            %                     prompts{kk} = prompts{kk} + accumarray([apu1 apu2],1,[detectors detectors],[],[],true);
            %                 end
            %                 if options.obtain_trues
            %                     trues_index = logical(Ltrues(tpoints(ll) + 1:tpoints(ll+1)));
            %                     trues_index = trues_index(inde);
            %                     if isempty(trues{kk})
            %                         trues{kk} = accumarray([apu1(trues_index) apu2(trues_index)],1,[detectors detectors],[],[],true);
            %                     else
            %                         trues{kk} = trues{kk} + accumarray([apu1(trues_index) apu2(trues_index)],1,[detectors detectors],[],[],true);
            %                     end
            %                 end
            %                 if options.store_scatter
            %                     trues_index = logical(Lscatter(tpoints(ll) + 1:tpoints(ll+1)));
            %                     trues_index = trues_index(inde);
            %                     if isempty(scatter{kk})
            %                         scatter{kk} = accumarray([apu1(trues_index) apu2(trues_index)],1,[detectors detectors],[],[],true);
            %                     else
            %                         scatter{kk} = scatter{kk} + accumarray([apu1(trues_index) apu2(trues_index)],1,[detectors detectors],[],[],true);
            %                     end
            %                 end
            %                 if options.store_randoms
            %                     trues_index = logical(Lrandoms(tpoints(ll) + 1:tpoints(ll+1)));
            %                     trues_index = trues_index(inde);
            %                     if isempty(randoms{kk})
            %                         randoms{kk} = accumarray([apu1(trues_index) apu2(trues_index)],1,[detectors detectors],[],[],true);
            %                     else
            %                         randoms{kk} = randoms{kk} + accumarray([apu1(trues_index) apu2(trues_index)],1,[detectors detectors],[],[],true);
            %                     end
            %                 end
            %                 ll = ll + 1;
            %             end
            %         end
            %     end
            % end
            if options.verbose
                disp(['File ' fnames(lk).name ' loaded'])
            end

        end
        clear Lrandoms apu2 Lscatter Ltrues apu1 trues_index M FOV CC i j k t t1 t2 t3 A S inde
        if source
            save([machine_name '_Ideal_image_coordinates_' name '_LMF.mat'],'C')
        end

    elseif options.use_ASCII
        %% ASCII data
        [ascii_ind, mSize] = get_ascii_indices(options.coincidence_mask);
        rsector_ind1 = ascii_ind.rsector_ind1;
        rsector_ind2 = ascii_ind.rsector_ind2;
        crs_ind1 = ascii_ind.crs_ind1;
        crs_ind2 = ascii_ind.crs_ind2;
        time_index = ascii_ind.time_index;
        source_index1 = ascii_ind.source_index1;
        source_index2 = ascii_ind.source_index2;
        % det_per_ring = options.det_per_ring;
        no_submodules = true;
        truesScatter = [true;true;true;true];
        if options.store_scatter && ascii_ind.scatter_index_cp1 == 0 && ascii_ind.scatter_index_cp2 == 0 ...
                && ascii_ind.scatter_index_cd1 == 0 && ascii_ind.scatter_index_cd2 == 0 ...
                && ascii_ind.scatter_index_rp1 == 0 && ascii_ind.scatter_index_rp2 == 0 ...
                && ascii_ind.scatter_index_rd1 == 0 && ascii_ind.scatter_index_rd2 == 0
            error('Store scatter selected, but no scatter data saved in coincidence mask')
        end
        if options.store_scatter && options.scatter_components(1) == 1 && (ascii_ind.scatter_index_cp1 == 0 || ascii_ind.scatter_index_cp2 == 0)
            warning('Compton scattering in the phantom selected, but no Compton scattering in the phantom selected in coincidence mask')
            options.scatter_components(1) = 0;
            truesScatter(1) = false;
        end
        if options.store_scatter && options.scatter_components(2) == 1 && (ascii_ind.scatter_index_cd1 == 0 || ascii_ind.scatter_index_cd2 == 0)
            warning('Compton scattering in the detector selected, but no Compton scattering in the detector selected in coincidence mask')
            options.scatter_components(2) = 0;
            truesScatter(2) = false;
        end
        if options.store_scatter && options.scatter_components(3) == 1 && (ascii_ind.scatter_index_rp1 == 0 || ascii_ind.scatter_index_rp2 == 0)
            warning('Rayleigh scattering in the phantom selected, but no Rayleigh scattering in the phantom selected in coincidence mask')
            options.scatter_components(3) = 0;
            truesScatter(3) = false;
        end
        if options.store_scatter && options.scatter_components(4) == 1 && (ascii_ind.scatter_index_rd1 == 0 || ascii_ind.scatter_index_rd2 == 0)
            warning('Rayleigh scattering in the detector selected, but no Rayleigh scattering in the detector selected in coincidence mask')
            options.scatter_components(4) = 0;
            truesScatter(4) = false;
        end
        if options.store_randoms && (ascii_ind.event_index1 == 0 || ascii_ind.event_index2 == 0)
            error('Store randoms selected, but event IDs not selected in coincidence mask')
        end
        if options.obtain_trues && ((ascii_ind.event_index1 == 0 || ascii_ind.event_index2 == 0) || (ascii_ind.scatter_index_cp1 == 0 || ascii_ind.scatter_index_cp2 == 0))
            error('Obtain trues selected, but event IDs and/or Compton scatter data in phantom not selected in coincidence mask')
        end


        % if partitions > 1
        %     save_string_raw = [machine_name '_measurements_' name '_' num2str(partitions) 'timepoints_for_total_of_' num2str(loppu - alku) 's_raw_ASCII.mat'];
        % else
        %     save_string_raw = [machine_name '_measurements_' name '_static_raw_ASCII.mat'];
        % end

        % Take coincidence data
        [~,~,ext] = fileparts(fpath);
        if strcmp(ext,'dat')
            fnames = dir(fpath);
        else
            fnames = dir([fpath '/*Coincidences*.dat']);
        end

        if size(fnames,1) == 0
            fpath = pwd;
            fpath = [fpath '/'];
            fnames = dir([fpath '*Coincidences*.dat']);
            if size(fnames,1) == 0
                warning('No ASCII (.dat) coincidence files were found. Input the ASCII output file (if you used a cluster, input the number 1 file).')
                [file, fpath] = uigetfile({'*.dat'},'Select the first GATE ASCII output file');
                if isequal(file, 0)
                    error('No file was selected!')
                end
                fnames = dir([fpath file(1:end-17) '*Coincidences*.dat']);
            end
        end

        if options.randoms_correction
            delay_names = dir([fpath '/*delay*.dat']);
            if size(delay_names,1) == 0
                dfpath = pwd;
                dfpath = [dfpath '/'];
                delay_names = dir([dfpath '*delay*.dat']);
                if size(delay_names,1) == 0
                    delay_names = dir([fpath file(1:end-17) '*delay*.dat']);
                    if size(delay_names,1) == 0
                        warning('No ASCII (.dat) delayed coincidence files were found. Input the ASCII delayed output file (if you used a cluster, input the number 1 file).')
                        [file, fpath] = uigetfile({'*.dat'},'Select the first GATE ASCII delayed output file');
                        if isequal(file, 0)
                            error('No file was selected!')
                        end
                        fnames = dir([fpath file(1:end-17) '*delay*.dat']);
                    end
                end
            end
        end
        % li = 1;

        if options.verbose
            disp('Data load started, this can take several minutes...')
        end

        % Go through all the files
        for lk = 1:length(fnames)

            % Use readmatrix if newer MATLAB is used, otherwise importdata
            if exist('OCTAVE_VERSION','builtin') == 0 && verLessThan('matlab','9.6') || exist('OCTAVE_VERSION','builtin') == 5
                M = importdata([fpath fnames(lk).name]);
                % Check for corrupted data
                if any(any(isnan(M)))
                    header = size(M,1);
                    warning(['Line ' num2str(header) ' corrupted, skipping.'])
                    B = importdata([fpath fnames(lk).name],' ', header);
                    while any(any(isnan(B.data)))
                        header = header + size(B.data,1);
                        M = [M; B.data(1:end-1,:)];
                        B = importdata([fpath fnames(lk).name],' ', header);
                        warning(['Line ' num2str(header) ' corrupted, skipping.'])
                    end
                    M = [M; B.data];
                    clear B
                end
            else
                M = readmatrix([fpath fnames(lk).name],'Encoding','UTF-8');
                if sum(isnan(M(:,end))) > 0 && sum(isnan(M(:,end))) > round(size(M,1)/2)
                    [rows, ~] = find(~isnan(M(:,end)));
                    for kg = 1 : length(rows)
                        warning(['Line ' num2str(rows(kg)) ' corrupted, skipping.'])
                        M(rows(kg),:) = [];
                    end
                else
                    if any(any(isnan(M)))
                        [rows, ~] = find(isnan(M));
                        rows = unique(rows);
                        for kg = 1 : length(rows)
                            warning(['Line ' num2str(rows(kg)) ' corrupted, skipping.'])
                            M(rows(kg),:) = [];
                        end
                    end
                end
            end
            if mSize - 6 == size(M,2)
                [ascii_ind, mSize] = get_ascii_indices(options.coincidence_mask, true);
            end
            % If no module indices are present (i.e. ECAT data or submodules are used)
            if ascii_ind.module_ind1 > 0 && sum(M(:,ascii_ind.module_ind1)) == 0
                ascii_ind.module_ind1 = 0;
                ascii_ind.module_ind2 = 0;
            end
            if ascii_ind.submodule_ind1 > 0 && sum(M(:,ascii_ind.submodule_ind1)) > 0
                no_submodules = false;
            end

            tIndex = [];
            if alku > 0
                if time_index == 0
                    error('Starting time is greater than zero, but no time data was found. Aborting!')
                end
                tIndex = M(:,time_index) < alku;
            end
            if loppu < (1e9-1) && ~isinf(options.end) && options.end < options.tot_time
                if time_index == 0
                    error('End time is smaller than total time, but no time data was found. Aborting!')
                end
                tIndex = logical(tIndex + M(:,time_index) > loppu);
            end
            if ~isempty(tIndex)
                M(tIndex, :) = [];
            end
            if Nt > 1
                timeI = [alku;alku + cumsum(partitions)];
                tIndex = ones(size(M,1),1,'uint16');
                for uu = 1 : Nt
                    test = M(:,time_index) < timeI(uu + 1) & M(:,time_index) >= timeI(uu);
                    if any(test)
                        alku = find(M(:,time_index) >= timeI(uu),1,'first');
                        loppu = find(M(:,time_index) >= timeI(uu),1,'last');
                        tIndex(alku:loppu) = uu;
                    end
                end
                % if isempty(find(M(:,time_index) > partitions(1),1,'last'))
                %     int_loc(1) = 0;
                % elseif isempty(find(M(1,time_index) > partitions,1,'last'))
                %     int_loc(1) = 1;
                % else
                %     int_loc(1) = find(M(1,time_index) > partitions,1,'last');
                % end
                % if isempty(find(M(:,time_index) < partitions(end),1,'first'))
                %     int_loc(2) = 0;
                % elseif isempty(find(M(end,time_index) < partitions,1,'first'))
                %     int_loc(2) = length(partitions) - 1;
                % else
                %     int_loc(2) = find(M(end,time_index) < partitions,1,'first') - 1;
                % end
                if max(M(:,time_index)) > max_time
                    max_time = max(M(:,time_index));
                end
                % timeI = double(M(:,time_index));
                % else
                %     int_loc(1) = 1;
                %     int_loc(2) = 1;
                %     timeI = 0;
            end

            if options.obtain_trues
                trues_index = false(size(M,1),1);
                if truesScatter(1) > 0 && truesScatter(2) > 0 && truesScatter(3) > 0 && truesScatter(4) > 0 ...
                        && ascii_ind.scatter_index_cd1 > 0 && ascii_ind.scatter_index_cd2 > 0 ...
                        && ascii_ind.scatter_index_rp1 > 0 && ascii_ind.scatter_index_rp2 > 0 ...
                        && ascii_ind.scatter_index_rd1 > 0 && ascii_ind.scatter_index_rd2 > 0
                    ind = (M(:,ascii_ind.event_index1) == M(:,ascii_ind.event_index2)) & (M(:,ascii_ind.scatter_index_cp1) == 0 & M(:,ascii_ind.scatter_index_cp2) == 0) & ...
                        (M(:,ascii_ind.scatter_index_cd1) == 0 & M(:,ascii_ind.scatter_index_cd2) == 0) & (M(:,ascii_ind.scatter_index_rp1) == 0 & M(:,ascii_ind.scatter_index_rp2) == 0) ...
                        & (M(:,ascii_ind.scatter_index_rd1) == 0 & M(:,ascii_ind.scatter_index_rd2) == 0);
                    if options.verbose && lk == 1
                        disp('Randoms, Compton scattered coincidences in the phantom and detector and Rayleigh scattered coincidences in the phantom and detector are NOT included in trues')
                    end
                elseif truesScatter(1) > 0 && truesScatter(2) > 0 && truesScatter(3) > 0 && truesScatter(4) == 0 ...
                        && ascii_ind.scatter_index_cd1 > 0 && ascii_ind.scatter_index_cd2 > 0 ...
                        && ascii_ind.scatter_index_rp1 > 0 && ascii_ind.scatter_index_rp2 > 0
                    ind = (M(:,ascii_ind.event_index1) == M(:,ascii_ind.event_index2)) & (M(:,ascii_ind.scatter_index_cp1) == 0 & M(:,ascii_ind.scatter_index_cp2) == 0) & ...
                        (M(:,ascii_ind.scatter_index_cd1) == 0 & M(:,ascii_ind.scatter_index_cd2) == 0) & (M(:,ascii_ind.scatter_index_rp1) == 0 & M(:,ascii_ind.scatter_index_rp2) == 0);
                    if options.verbose && lk == 1
                        disp('Randoms, Compton scattered coincidences in the phantom and detector and Rayleigh scattered coincidences in the phantom are NOT included in trues')
                    end
                elseif truesScatter(1) > 0 && truesScatter(2) > 0 && truesScatter(3) == 0 && truesScatter(4) == 0 ...
                        && ascii_ind.scatter_index_cd1 > 0 && ascii_ind.scatter_index_cd2 > 0
                    ind = (M(:,ascii_ind.event_index1) == M(:,ascii_ind.event_index2)) & (M(:,ascii_ind.scatter_index_cp1) == 0 & M(:,ascii_ind.scatter_index_cp2) == 0) & ...
                        (M(:,ascii_ind.scatter_index_cd1) == 0 & M(:,ascii_ind.scatter_index_cd2) == 0);
                    if options.verbose && lk == 1
                        disp('Randoms, Compton scattered coincidences in the phantom and detector are NOT included in trues')
                    end
                elseif truesScatter(1) > 0 && truesScatter(2) == 0 && truesScatter(3) == 0 && truesScatter(4) == 0
                    ind = (M(:,ascii_ind.event_index1) == M(:,ascii_ind.event_index2)) & (M(:,ascii_ind.scatter_index_cp1) == 0);
                    if options.verbose && lk == 1
                        disp('Randoms and Compton scattered coincidences in the phantom are NOT included in trues')
                    end
                elseif truesScatter(1) > 0 && truesScatter(2) > 0 && truesScatter(3) == 0 && truesScatter(4) > 0 ...
                        && ascii_ind.scatter_index_cd1 > 0 && ascii_ind.scatter_index_cd2 > 0 ...
                        && ascii_ind.scatter_index_rd1 > 0 && ascii_ind.scatter_index_rd2 > 0
                    ind = (M(:,ascii_ind.event_index1) == M(:,ascii_ind.event_index2)) & (M(:,ascii_ind.scatter_index_cp1) == 0 & M(:,ascii_ind.scatter_index_cp2) == 0) & ...
                        (M(:,ascii_ind.scatter_index_cd1) == 0 & M(:,ascii_ind.scatter_index_cd2) == 0) ...
                        & (M(:,ascii_ind.scatter_index_rd1) == 0 & M(:,ascii_ind.scatter_index_rd2) == 0);
                    if options.verbose && lk == 1
                        disp('Randoms, Compton scattered coincidences in the phantom and detector and Rayleigh scattered coincidences in the detector are NOT included in trues')
                    end
                elseif truesScatter(1) > 0 && truesScatter(2) == 0 && truesScatter(3) > 0 && truesScatter(4) > 0 ...
                        && ascii_ind.scatter_index_rp1 > 0 && ascii_ind.scatter_index_rp2 > 0 ...
                        && ascii_ind.scatter_index_rd1 > 0 && ascii_ind.scatter_index_rd2 > 0
                    ind = (M(:,ascii_ind.event_index1) == M(:,ascii_ind.event_index2)) & (M(:,ascii_ind.scatter_index_cp1) == 0 & M(:,ascii_ind.scatter_index_cp2) == 0) & ...
                        (M(:,ascii_ind.scatter_index_rp1) == 0 & M(:,ascii_ind.scatter_index_rp2) == 0) ...
                        & (M(:,ascii_ind.scatter_index_rd1) == 0 & M(:,ascii_ind.scatter_index_rd2) == 0);
                    if options.verbose && lk == 1
                        disp('Randoms, Compton scattered coincidences in the phantom and Rayleigh scattered coincidences in the phantom and detector are NOT included in trues')
                    end
                elseif truesScatter(1) > 0 && truesScatter(2) == 0 && truesScatter(3) > 0 && truesScatter(4) == 0 ...
                        && ascii_ind.scatter_index_rp1 > 0 && ascii_ind.scatter_index_rp2 > 0
                    ind = (M(:,ascii_ind.event_index1) == M(:,ascii_ind.event_index2)) & (M(:,ascii_ind.scatter_index_cp1) == 0 & M(:,ascii_ind.scatter_index_cp2) == 0) & ...
                        (M(:,ascii_ind.scatter_index_rp1) == 0 & M(:,ascii_ind.scatter_index_rp2) == 0);
                    if options.verbose && lk == 1
                        disp('Randoms, Compton scattered coincidences in the phantom and Rayleigh scattered coincidences in the phantom are NOT included in trues')
                    end
                elseif truesScatter(1) > 0 && truesScatter(2) == 0 && truesScatter(3) == 0 && truesScatter(4) > 0 ...
                        && ascii_ind.scatter_index_rd1 > 0 && ascii_ind.scatter_index_rd2 > 0
                    ind = (M(:,ascii_ind.event_index1) == M(:,ascii_ind.event_index2)) & (M(:,ascii_ind.scatter_index_cp1) == 0 & M(:,ascii_ind.scatter_index_cp2) == 0) & ...
                        (M(:,ascii_ind.scatter_index_rd1) == 0 & M(:,ascii_ind.scatter_index_rd2) == 0);
                    if options.verbose && lk == 1
                        disp('Randoms, Compton scattered coincidences in the phantom and Rayleigh scattered coincidences in the detector are NOT included in trues')
                    end
                end
                trues_index(ind) = true;
            end
            if options.store_scatter
                scatter_index = false(size(M,1),1);
                if options.scatter_components(1) > 0 && ascii_ind.scatter_index_cp1 > 0 && ascii_ind.scatter_index_cp2
                    ind = (M(:,ascii_ind.scatter_index_cp1) >= options.scatter_components(1) | ...
                        M(:,ascii_ind.scatter_index_cp2) >= options.scatter_components(1)) & (M(:,ascii_ind.event_index1) == M(:,ascii_ind.event_index2));
                    scatter_index(ind) = true;
                    if options.verbose && lk == 1
                        disp('Compton scatter in the phantom will be stored')
                    end
                end
                if options.scatter_components(2) > 0 && ascii_ind.scatter_index_cd1 > 0 && ascii_ind.scatter_index_cd2
                    ind = (M(:,ascii_ind.scatter_index_cd1) >= options.scatter_components(2) | ...
                        M(:,ascii_ind.scatter_index_cd2) >= options.scatter_components(2)) & (M(:,ascii_ind.event_index1) == M(:,ascii_ind.event_index2));
                    scatter_index(ind) = true;
                    if options.verbose && lk == 1
                        disp('Compton scatter in the detector will be stored')
                    end
                end
                if options.scatter_components(3) > 0 && ascii_ind.scatter_index_rp1 > 0 && ascii_ind.scatter_index_rp2
                    ind = (M(:,ascii_ind.scatter_index_rp1) >= options.scatter_components(3) | ...
                        M(:,ascii_ind.scatter_index_rp2) >= options.scatter_components(3)) & (M(:,ascii_ind.event_index1) == M(:,ascii_ind.event_index2));
                    scatter_index(ind) = true;
                    if options.verbose && lk == 1
                        disp('Rayleigh scatter in the phantom will be stored')
                    end
                end
                if options.scatter_components(4) > 0 && ascii_ind.scatter_index_rd1 > 0 && ascii_ind.scatter_index_rd2
                    ind = (M(:,ascii_ind.scatter_index_rd1) >= options.scatter_components(4) | ...
                        M(:,ascii_ind.scatter_index_rd2) >= options.scatter_components(4)) & (M(:,ascii_ind.event_index1) == M(:,ascii_ind.event_index2));
                    scatter_index(ind) = true;
                    if options.verbose && lk == 1
                        disp('Rayleigh scatter in the detector will be stored')
                    end
                end
            end
            if options.store_randoms
                randoms_index = M(:,ascii_ind.event_index1) ~= M(:,ascii_ind.event_index2);
            end
            clear ind

            % The number of crystal ring of each coincidence event (e.g. the first single hits
            % a detector on crystal ring 5 and second a detector on crystal ring 10)
            % [0 rings - 1]
            % if int_loc(1) > 0
            % No modules
            if options.nLayers > 1
                layer1 = uint8(M(:,ascii_ind.layer_ind2));
                layer2 = uint8(M(:,ascii_ind.layer_ind1));
            else
                layer1 = uint8(0);
                layer2 = uint8(0);
            end
            if (ascii_ind.module_ind1 == 0 || ascii_ind.module_ind2 == 0) && options.linear_multip > 1 && no_submodules
                ring_number1 = uint16(floor(M(:,rsector_ind1) / blocks_per_ring) * cryst_per_block_z + floor(M(:,crs_ind1)/cryst_per_block));
                ring_number2 = uint16(floor(M(:,rsector_ind2) / blocks_per_ring) * cryst_per_block_z + floor(M(:,crs_ind2)/cryst_per_block));
                % Only a single ring
            elseif options.rings == 1
                ring_number1 = zeros(size(M,1),1,'uint16');
                ring_number2 = zeros(size(M,1),1,'uint16');
                % No modules and no repeated axial rings
            elseif (ascii_ind.module_ind1 == 0 || ascii_ind.module_ind2 == 0) && options.linear_multip == 1 && no_submodules
                ring_number1 = uint16(M(:,rsector_ind1));
                ring_number2 = uint16(M(:,rsector_ind2));
                % No repeated axial rings (modules)
            elseif options.linear_multip == 1 && no_submodules
                ring_number1 = uint16(M(:,ascii_ind.module_ind1));
                ring_number2 = uint16(M(:,ascii_ind.module_ind2));
                % No repeated axial rings (submodules)
            elseif options.linear_multip == 1 && ~no_submodules
                ring_number1 = uint16(M(:,ascii_ind.submodule_ind1));
                ring_number2 = uint16(M(:,ascii_ind.submodule_ind2));
            elseif ~no_submodules
                if transaxial_multip > 1
                    ring_number1 = uint16(mod(floor(M(:,ascii_ind.submodule_ind1) / transaxial_multip), options.linear_multip) * cryst_per_block_z + floor(M(:,crs_ind1) / cryst_per_block));
                    ring_number2 = uint16(mod(floor(M(:,ascii_ind.submodule_ind2) / transaxial_multip), options.linear_multip) * cryst_per_block_z + floor(M(:,crs_ind2) / cryst_per_block));
                else
                    ring_number1 = uint16(mod(M(:,ascii_ind.submodule_ind1), options.linear_multip) * cryst_per_block_z + floor(M(:,crs_ind1) / cryst_per_block));
                    ring_number2 = uint16(mod(M(:,ascii_ind.submodule_ind2), options.linear_multip) * cryst_per_block_z + floor(M(:,crs_ind2) / cryst_per_block));
                end
                % "Normal" case
            else
                if transaxial_multip > 1
                    if options.axial_multip > 1
                        ring_number1 = uint16(floor(M(:,rsector_ind1) / blocks_per_ring) * cryst_per_block_z * options.axial_multip + ((floor(M(:,ascii_ind.module_ind1) / transaxial_multip)) * cryst_per_block + (floor(M(:,crs_ind1) / cryst_per_block))));
                        ring_number2 = uint16(floor(M(:,rsector_ind2) / blocks_per_ring) * cryst_per_block_z * options.axial_multip + ((floor(M(:,ascii_ind.module_ind2) / transaxial_multip)) * cryst_per_block + (floor(M(:,crs_ind2) / cryst_per_block))));
                    else
                        ring_number1 = uint16(mod(floor(M(:,ascii_ind.module_ind1) / transaxial_multip), options.linear_multip) * cryst_per_block_z + floor(M(:,crs_ind1) / cryst_per_block));
                        ring_number2 = uint16(mod(floor(M(:,ascii_ind.module_ind2) / transaxial_multip), options.linear_multip) * cryst_per_block_z + floor(M(:,crs_ind2) / cryst_per_block));
                    end
                else
                    if options.axial_multip > 1
                        ring_number1 = uint16(floor(M(:,rsector_ind1) / blocks_per_ring) * cryst_per_block_z * options.axial_multip + ((floor(M(:,ascii_ind.module_ind1))) * cryst_per_block + (floor(M(:,crs_ind1) / cryst_per_block))));
                        ring_number2 = uint16(floor(M(:,rsector_ind2) / blocks_per_ring) * cryst_per_block_z * options.axial_multip + ((floor(M(:,ascii_ind.module_ind2))) * cryst_per_block + (floor(M(:,crs_ind2) / cryst_per_block))));
                    else
                        ring_number1 = uint16(mod(M(:,ascii_ind.module_ind1), options.linear_multip) * cryst_per_block_z + floor(M(:,crs_ind1) / cryst_per_block));
                        ring_number2 = uint16(mod(M(:,ascii_ind.module_ind2), options.linear_multip) * cryst_per_block_z + floor(M(:,crs_ind2) / cryst_per_block));
                    end
                end
            end

            % detector number of the single at the above ring (e.g. first single hits a
            % detector number 54 on ring 5)
            % [0 det_per_ring - 1]
            if transaxial_multip > 1
                if no_submodules
                    ring_pos1 = uint16(mod(M(:,rsector_ind1), blocks_per_ring) * cryst_per_block * transaxial_multip + mod(M(:,crs_ind1), cryst_per_block) + mod(M(:,ascii_ind.module_ind1), transaxial_multip) * cryst_per_block);
                    ring_pos2 = uint16(mod(M(:,rsector_ind2), blocks_per_ring) * cryst_per_block * transaxial_multip + mod(M(:,crs_ind2), cryst_per_block) + mod(M(:,ascii_ind.module_ind2), transaxial_multip) * cryst_per_block);
                else
                    ring_pos1 = uint16(mod(M(:,rsector_ind1), blocks_per_ring) * cryst_per_block * transaxial_multip + mod(M(:,crs_ind1), cryst_per_block) + mod(M(:,ascii_ind.submodule_ind1), transaxial_multip) * cryst_per_block);
                    ring_pos2 = uint16(mod(M(:,rsector_ind2), blocks_per_ring) * cryst_per_block * transaxial_multip + mod(M(:,crs_ind2), cryst_per_block) + mod(M(:,ascii_ind.submodule_ind2), transaxial_multip) * cryst_per_block);
                end
            else
                ring_pos1 = uint16(mod(M(:,rsector_ind1), blocks_per_ring) * cryst_per_block + mod(M(:,crs_ind1), cryst_per_block));
                ring_pos2 = uint16(mod(M(:,rsector_ind2), blocks_per_ring) * cryst_per_block + mod(M(:,crs_ind2), cryst_per_block));
            end

            % if options.store_raw_data
            %     % detector number (MATLAB numbering) of each single
            %     % [1 det_per_ring * rings]
            %     if partitions == 1
            %         LL1 = ring_number1*uint16(det_per_ring) + ring_pos1 + 1;
            %         LL2 = ring_number2*uint16(det_per_ring) + ring_pos2 + 1;
            %         LL3 = LL2;
            %         LL2(LL2 > LL1) = LL1(LL2 > LL1);
            %         LL1(LL3 > LL1) = LL3(LL3 > LL1);
            %         clear LL3
            %         if sum(prompts(:)) == 0
            %             prompts = prompts + accumarray([LL1 LL2],lisays,[detectors detectors],@(x) sum(x,'native'));
            %             if max(prompts(:)) >= 65535
            %                 lisays = uint32(1);
            %                 prompts = accumarray([LL1 LL2],lisays,[detectors detectors],@(x) sum(x,'native'));
            %             end
            %         else
            %             prompts = prompts + accumarray([LL1 LL2],lisays,[detectors detectors],@(x) sum(x,'native'));
            %         end
            %         if options.obtain_trues
            %             if any(trues_index)
            %                 trues = trues + accumarray([LL1(trues_index) LL2(trues_index)],lisays,[detectors detectors],@(x) sum(x,'native'));
            %             end
            %         end
            %         if options.store_scatter
            %             if any(scatter_index)
            %                 scatter = scatter + accumarray([LL1(scatter_index) LL2(scatter_index)],lisays,[detectors detectors],@(x) sum(x,'native'));
            %             end
            %         end
            %         if options.store_randoms
            %             if any(randoms_index)
            %                 randoms = randoms + accumarray([LL1(randoms_index) LL2(randoms_index)],lisays,[detectors detectors],@(x) sum(x,'native'));
            %             end
            %         end
            %         li = li + length(LL1);
            %     else
            %         LL1 = ring_number1*uint16(det_per_ring) + ring_pos1 + 1;
            %         LL2 = ring_number2*uint16(det_per_ring) + ring_pos2 + 1;
            %         LL3 = LL2;
            %         LL2(LL2 > LL1) = LL1(LL2 > LL1);
            %         LL1(LL3 > LL1) = LL3(LL3 > LL1);
            %         clear LL3
            %         ll = 1;
            %         for kk = int_loc(1) : int_loc(2)
            %             index = find(M(:,time_index)<partitions(kk+1),1,'last');
            %             if isempty(prompts{kk})
            %                 prompts{kk} = accumarray([LL1(ll:index) LL2(ll:index)],1,[detectors detectors],[],[],true);
            %             else
            %                 prompts{kk} = prompts{kk} + accumarray([LL1(ll:index) LL2(ll:index)],1,[detectors detectors],[],[],true);
            %             end
            %             if options.obtain_trues
            %                 apu_trues = trues_index(ll:index);
            %                 L1trues = LL1(ll:index);
            %                 L1trues = L1trues(apu_trues);
            %                 L2trues = LL2(ll:index);
            %                 L2trues = L2trues(apu_trues);
            %                 if isempty(trues{kk})
            %                     trues{kk} = accumarray([L1trues L2trues],1,[detectors detectors],[],[],true);
            %                 else
            %                     trues{kk} = trues{kk} + accumarray([L1trues L2trues],1,[detectors detectors],[],[],true);
            %                 end
            %             end
            %             if options.store_scatter
            %                 apu_scatter = scatter_index(ll:index);
            %                 L1scatter = LL1(ll:index);
            %                 L1scatter = L1scatter(apu_scatter);
            %                 L2scatter = LL2(ll:index);
            %                 L2scatter = L2scatter(apu_scatter);
            %                 if isempty(scatter{kk})
            %                     scatter{kk} = accumarray([L1scatter L2scatter],1,[detectors detectors],[],[],true);
            %                 else
            %                     scatter{kk} = scatter{kk} + accumarray([L1scatter L2scatter],1,[detectors detectors],[],[],true);
            %                 end
            %             end
            %             if options.store_randoms
            %                 apu_randoms = randoms_index(ll:index);
            %                 L1randoms = LL1(ll:index);
            %                 L1randoms = L1randoms(apu_randoms);
            %                 L2randoms = LL2(ll:index);
            %                 L2randoms = L2randoms(apu_randoms);
            %                 if isempty(randoms{kk})
            %                     randoms{kk} = accumarray([L1randoms L2randoms],1,[detectors detectors],[],[],true);
            %                 else
            %                     randoms{kk} = randoms{kk} + accumarray([L1randoms L2randoms],1,[detectors detectors],[],[],true);
            %                 end
            %             end
            %             ll = index + 1;
            %         end
            %     end
            % end

            if TOF
                TOF_data = (M(:,ascii_ind.time_index2) - M(:,ascii_ind.time_index));
                TOF_data(ring_pos2 > ring_pos1) = -TOF_data(ring_pos2 > ring_pos1);
                [bins, discard] = FormTOFBins(options, TOF_data);
                ring_number1(discard) = [];
                ring_number2(discard) = [];
                ring_pos1(discard) = [];
                ring_pos2(discard) = [];
                if Nt > 1
                    tIndex(discard) = [];
                end
                if options.obtain_trues
                    trues_index(discard) = [];
                end
                if options.store_scatter
                    scatter_index(discard) = [];
                end
                if options.store_randoms
                    randoms_index(discard) = [];
                end
                clear discard
            else
                bins = 0;
            end

            if options.useIndexBasedReconstruction
                trIndices = [trIndices,[ring_pos1;ring_pos2]];
                axIndices = [axIndices,[ring_number1;ring_number2]];
            else
                if ~options.use_raw_data
                    if exist('OCTAVE_VERSION','builtin') == 0% && verLessThan('matlab','9.4')
                        [raw_SinM, SinTrues, SinScatter, SinRandoms] = createSinogramASCII(ring_pos1, ring_pos2, ring_number1, ring_number2, trues_index, scatter_index, ... % 7
                            randoms_index, sinoSize, (options.Ndist), (options.Nang), (options.ring_difference), (options.span), ... % 13
                            uint32(cumsum(options.segment_table)), sinoSize * uint64(options.TOF_bins), tIndex, Nt, ... % 17
                            (options.det_per_ring), int32(options.rings), uint16(bins), raw_SinM, SinTrues, SinScatter, SinRandoms, (options.ndist_side),... % 26
                            (options.det_w_pseudo), int32(sum(options.pseudot)), (options.cryst_per_block), options.nLayers, layer1, layer2); % 32
                    elseif exist('OCTAVE_VERSION','builtin') == 5
                        [raw_SinM, SinTrues, SinScatter, SinRandoms] = createSinogramASCIIOct(ring_pos1, ring_pos2, ring_number1, ring_number2, trues_index, scatter_index, ... % 7
                            randoms_index, sinoSize, (options.Ndist), (options.Nang), (options.ring_difference), (options.span), ... % 13
                            uint32(cumsum(options.segment_table)), sinoSize * uint64(options.TOF_bins), tIndex, Nt, ... % 17
                            (options.det_per_ring), int32(options.rings), uint16(bins), raw_SinM, SinTrues, SinScatter, SinRandoms, (options.ndist_side),... % 26
                            (options.det_w_pseudo), int32(sum(options.pseudot)), (options.cryst_per_block), options.nLayers, layer1, layer2); % 32
                        % else
                        %     [raw_SinM, SinTrues, SinScatter, SinRandoms] = createSinogramASCIICPP(vali, ring_pos1, ring_pos2, ring_number1, ring_number2, trues_index, scatter_index, ...
                        %         randoms_index, sinoSize, uint32(options.Ndist), uint32(options.Nang), uint32(options.ring_difference), uint32(options.span), ...
                        %         uint32(cumsum(options.segment_table)), sinoSize * uint64(options.TOF_bins), timeI, uint64(options.partitions), ...
                        %         alku, int32(options.det_per_ring), int32(options.rings), uint16(bins), raw_SinM(:), SinTrues(:), SinScatter(:), SinRandoms(:), int32(options.ndist_side),...
                        %         int32(options.det_w_pseudo), int32(sum(options.pseudot)), int32(options.cryst_per_block), options.nLayers, layer1, layer2);
                    end
                end

                if store_coordinates && ascii_ind.world_index1 > 0 && ascii_ind.world_index2 > 0
                    if Nt == 1
                        coordinate = [coordinate, M(:,ascii_ind.world_index1)';M(:,ascii_ind.world_index1+1)';M(:,ascii_ind.world_index1+2)'; M(:,ascii_ind.world_index2)'; M(:,ascii_ind.world_index2+1)'; M(:,ascii_ind.world_index2+2)'];
                        % temp = [M(:,ascii_ind.world_index1), M(:,ascii_ind.world_index2)];
                        % x_coordinate = [x_coordinate; temp];
                        % temp = [M(:,ascii_ind.world_index1+1), M(:,ascii_ind.world_index2+1)];
                        % y_coordinate = [y_coordinate; temp];
                        % temp = [M(:,ascii_ind.world_index1+2), M(:,ascii_ind.world_index2+2)];
                        % z_coordinate = [z_coordinate; temp];
                    else
                        uind = unique(tIndex);
                        x1 = M(:,ascii_ind.world_index1);
                        x2 = M(:,ascii_ind.world_index2);
                        y1 = M(:,ascii_ind.world_index1 + 1);
                        y2 = M(:,ascii_ind.world_index2 + 1);
                        z1 = M(:,ascii_ind.world_index1 + 2);
                        z2 = M(:,ascii_ind.world_index2 + 2);
                        for uu = 1 : numel(uind)
                            temp = [x1(tIndex == uind(uu))';y1(tIndex == uind(uu))';z1(tIndex == uind(uu))';x2(tIndex == uind(uu))';y2(tIndex == uind(uu))';z2(tIndex == uind(uu))'];
                            coordinate{uu} = [coordinate{uu}, temp];
                            % x_coordinate{uind(uu)} = [x_coordinate{uind(uu)}; temp];
                            % y_coordinate{uind(uu)} = [y_coordinate{uind(uu)}; temp];
                            % z_coordinate{uind(uu)} = [z_coordinate{uind(uu)}; temp];
                        end
                    end
                end
            end

            % An ideal image can be formed with this
            % Takes the columns containing the source locations for both singles
            if source_index1 ~= 0 && ~isempty(source_index1) && source_index2 ~= 0 && ~isempty(source_index2) && source

                if Nt > 1
                    ut = unique(tIndex);
                    for uu = 1 : numel(ut)
                        ll = find(tIndex == ut(uu),1,'first');
                        index = find(tIndex == ut(uu),1,'last');
                        tt = ut(uu);
                        [C, SC, RA] = formSourceImage(options, S, trues_index, scatter_index, randoms_index, C, SC, RA, ll, index, tt);
                    end
                else
                    S = single(M(:,[source_index1:source_index1+2 source_index2:source_index2+2]));
                    [C, SC, RA] = formSourceImage(options, S, trues_index, scatter_index, randoms_index, C, SC, RA);
                end
            end
            % end

            if options.verbose
                disp(['File ' fnames(lk).name ' loaded'])
            end

            if options.randoms_correction && length(delay_names) >= lk
                if exist('OCTAVE_VERSION','builtin') == 0 && verLessThan('matlab','9.6') || exist('OCTAVE_VERSION','builtin') == 5
                    M = importdata([fpath delay_names(lk).name]);
                    % Check for corrupted data
                    if any(any(isnan(M)))
                        header = size(M,1);
                        warning(['Line ' num2str(header) ' corrupted, skipping.'])
                        B = importdata([fpath delay_names(lk).name],' ', header);
                        while any(any(isnan(B.data)))
                            header = header + size(B.data,1);
                            M = [M; B.data(1:end-1,:)];
                            B = importdata([fpath delay_names(lk).name],' ', header);
                            warning(['Line ' num2str(header) ' corrupted, skipping.'])
                        end
                        M = [M; B.data];
                        clear B
                    end
                else
                    M = readmatrix([fpath delay_names(lk).name]);
                    % Check for corrupted data
                    if sum(isnan(M(:,end))) > 0 && sum(isnan(M(:,end))) > round(size(M,1)/2)
                        [rows, ~] = find(~isnan(M(:,end)));
                        for kg = 1 : length(rows)
                            warning(['Line ' num2str(rows(kg)) ' corrupted, skipping.'])
                            M(rows(kg),:) = [];
                        end
                    else
                        if any(any(isnan(M))) > 0
                            [rows, ~] = find(isnan(M));
                            rows = unique(rows);
                            for kg = 1 : length(rows)
                                warning(['Line ' num2str(rows(kg)) ' corrupted, skipping.'])
                                M(rows(kg),:) = [];
                            end
                        end
                    end
                end
                tIndex = [];
                if alku > 0
                    tIndex = M(:,time_index) < alku;
                end
                if loppu < (1e9-1) && ~isinf(options.end) && options.end < options.tot_time
                    tIndex = logical(tIndex + M(:,time_index) > loppu);
                end
                if ~isempty(tIndex)
                    M(tIndex, :) = [];
                end
                if Nt > 1
                    timeI = [alku;alku + cumsum(partitions)];
                    tIndex = ones(size(M,1),1,'uint16');
                    for uu = 1 : Nt
                        test = M(:,time_index) < timeI(uu + 1) & M(:,time_index) >= timeI(uu);
                        if any(test)
                            alku = find(M(:,time_index) >= timeI(uu),1,'first');
                            loppu = find(M(:,time_index) >= timeI(uu),1,'last');
                            tIndex(alku:loppu) = uu;
                        end
                    end
                end

                % No modules
                if (ascii_ind.module_ind1 == 0 || ascii_ind.module_ind2 == 0) && options.linear_multip > 1 && no_submodules
                    ring_number1 = uint16(floor(M(:,rsector_ind1) / blocks_per_ring) * cryst_per_block_z + floor(M(:,crs_ind1)/cryst_per_block));
                    ring_number2 = uint16(floor(M(:,rsector_ind2) / blocks_per_ring) * cryst_per_block_z + floor(M(:,crs_ind2)/cryst_per_block));
                    % Only a single ring
                elseif options.rings == 1
                    ring_number1 = zeros(size(M,1),1,'uint16');
                    ring_number2 = zeros(size(M,1),1,'uint16');
                    % No modules and no repeated axial rings
                elseif (ascii_ind.module_ind1 == 0 || ascii_ind.module_ind2 == 0) && options.linear_multip == 1 && no_submodules
                    ring_number1 = uint16(M(:,rsector_ind1));
                    ring_number2 = uint16(M(:,rsector_ind2));
                    % No repeated axial rings (modules)
                elseif options.linear_multip == 1 && no_submodules
                    ring_number1 = uint16(M(:,ascii_ind.module_ind1));
                    ring_number2 = uint16(M(:,ascii_ind.module_ind2));
                    % No repeated axial rings (submodules)
                elseif options.linear_multip == 1 && ~no_submodules
                    ring_number1 = uint16(M(:,ascii_ind.submodule_ind1));
                    ring_number2 = uint16(M(:,ascii_ind.submodule_ind2));
                elseif ~no_submodules
                    if transaxial_multip > 1
                        ring_number1 = uint16(mod(floor(M(:,ascii_ind.submodule_ind1) / transaxial_multip), options.linear_multip) * cryst_per_block_z + floor(M(:,crs_ind1) / cryst_per_block));
                        ring_number2 = uint16(mod(floor(M(:,ascii_ind.submodule_ind2) / transaxial_multip), options.linear_multip) * cryst_per_block_z + floor(M(:,crs_ind2) / cryst_per_block));
                    else
                        ring_number1 = uint16(mod(M(:,ascii_ind.submodule_ind1), options.linear_multip) * cryst_per_block_z + floor(M(:,crs_ind1) / cryst_per_block));
                        ring_number2 = uint16(mod(M(:,ascii_ind.submodule_ind2), options.linear_multip) * cryst_per_block_z + floor(M(:,crs_ind2) / cryst_per_block));
                    end
                    % "Normal" case
                else
                    if transaxial_multip > 1
                        if options.axial_multip > 1
                            ring_number1 = uint16(floor(M(:,rsector_ind1) / blocks_per_ring) * cryst_per_block_z * options.axial_multip + ((floor(M(:,ascii_ind.module_ind1) / transaxial_multip)) * cryst_per_block + (floor(M(:,crs_ind1) / cryst_per_block))));
                            ring_number2 = uint16(floor(M(:,rsector_ind2) / blocks_per_ring) * cryst_per_block_z * options.axial_multip + ((floor(M(:,ascii_ind.module_ind2) / transaxial_multip)) * cryst_per_block + (floor(M(:,crs_ind2) / cryst_per_block))));
                        else
                            ring_number1 = uint16(mod(floor(M(:,ascii_ind.module_ind1) / transaxial_multip), options.linear_multip) * cryst_per_block_z + floor(M(:,crs_ind1) / cryst_per_block));
                            ring_number2 = uint16(mod(floor(M(:,ascii_ind.module_ind2) / transaxial_multip), options.linear_multip) * cryst_per_block_z + floor(M(:,crs_ind2) / cryst_per_block));
                        end
                    else
                        if options.axial_multip > 1
                            ring_number1 = uint16(floor(M(:,rsector_ind1) / blocks_per_ring) * cryst_per_block_z * options.axial_multip + ((floor(M(:,ascii_ind.module_ind1))) * cryst_per_block + (floor(M(:,crs_ind1) / cryst_per_block))));
                            ring_number2 = uint16(floor(M(:,rsector_ind2) / blocks_per_ring) * cryst_per_block_z * options.axial_multip + ((floor(M(:,ascii_ind.module_ind2))) * cryst_per_block + (floor(M(:,crs_ind2) / cryst_per_block))));
                        else
                            ring_number1 = uint16(mod(M(:,ascii_ind.module_ind1), options.linear_multip) * cryst_per_block_z + floor(M(:,crs_ind1) / cryst_per_block));
                            ring_number2 = uint16(mod(M(:,ascii_ind.module_ind2), options.linear_multip) * cryst_per_block_z + floor(M(:,crs_ind2) / cryst_per_block));
                        end
                    end
                end

                % detector number of the single at the above ring (e.g. first single hits a
                % detector number 54 on ring 5)
                % [0 det_per_ring - 1]
                if transaxial_multip > 1
                    if no_submodules
                        ring_pos1 = uint16(mod(M(:,rsector_ind1), blocks_per_ring) * cryst_per_block * transaxial_multip + mod(M(:,crs_ind1), cryst_per_block) + mod(M(:,ascii_ind.module_ind1), transaxial_multip) * cryst_per_block);
                        ring_pos2 = uint16(mod(M(:,rsector_ind2), blocks_per_ring) * cryst_per_block * transaxial_multip + mod(M(:,crs_ind2), cryst_per_block) + mod(M(:,ascii_ind.module_ind2), transaxial_multip) * cryst_per_block);
                    else
                        ring_pos1 = uint16(mod(M(:,rsector_ind1), blocks_per_ring) * cryst_per_block * transaxial_multip + mod(M(:,crs_ind1), cryst_per_block) + mod(M(:,ascii_ind.submodule_ind1), transaxial_multip) * cryst_per_block);
                        ring_pos2 = uint16(mod(M(:,rsector_ind2), blocks_per_ring) * cryst_per_block * transaxial_multip + mod(M(:,crs_ind2), cryst_per_block) + mod(M(:,ascii_ind.submodule_ind2), transaxial_multip) * cryst_per_block);
                    end
                else
                    ring_pos1 = uint16(mod(M(:,rsector_ind1), blocks_per_ring) * cryst_per_block + mod(M(:,crs_ind1), cryst_per_block));
                    ring_pos2 = uint16(mod(M(:,rsector_ind2), blocks_per_ring) * cryst_per_block + mod(M(:,crs_ind2), cryst_per_block));
                end
                if options.nLayers > 1
                    layer1 = uint8(M(:,ascii_ind.layer_ind1));
                    layer2 = uint8(M(:,ascii_ind.layer_ind2));
                else
                    layer1 = uint8(0);
                    layer2 = uint8(0);
                end

                if options.useIndexBasedReconstruction
                    DtrIndices = [DtrIndices,[ring_pos1;ring_pos2]];
                    DaxIndices = [DaxIndices,[ring_number1;ring_number2]];
                else
                    if ~options.use_raw_data
                        if exist('OCTAVE_VERSION','builtin') == 0 %&& verLessThan('matlab','9.4')
                            [SinD, ~, ~, ~] = createSinogramASCII(ring_pos1, ring_pos2, ring_number1, ring_number2, [], [], ...
                                [], sinoSize, (options.Ndist), (options.Nang), (options.ring_difference), (options.span), ...
                                uint32(cumsum(options.segment_table)), sinoSize * uint64(1), tIndex, Nt, ...
                                (options.det_per_ring), (options.rings), uint16([]), SinD, [], [], [], (options.ndist_side),...
                                (options.det_w_pseudo), int32(sum(options.pseudot)), (options.cryst_per_block), options.nLayers, layer1, layer2);
                        elseif exist('OCTAVE_VERSION','builtin') == 5
                            [SinD, ~, ~, ~] = createSinogramASCIIOct(ring_pos1, ring_pos2, ring_number1, ring_number2, [], [], ...
                                [], sinoSize, (options.Ndist), (options.Nang), (options.ring_difference), (options.span), ...
                                uint32(cumsum(options.segment_table)), sinoSize * uint64(1), tIndex, Nt, ...
                                (options.det_per_ring), (options.rings), uint16([]), SinD, [], [], [], (options.ndist_side),...
                                (options.det_w_pseudo), int32(sum(options.pseudot)), (options.cryst_per_block), options.nLayers, layer1, layer2);
                            % else
                            %     [SinD, ~, ~, ~] = createSinogramASCIICPP(vali, ring_pos1, ring_pos2, ring_number1, ring_number2, logical([]), logical([]), ...
                            %         logical([]), sinoSize, uint32(options.Ndist), uint32(options.Nang), uint32(options.ring_difference), uint32(options.span), ...
                            %         uint32(cumsum(options.segment_table)), sinoSize * uint64(1), M(:,time_index), uint64(options.partitions), ...
                            %         alku, int32(options.det_per_ring), int32(options.rings), uint16([]), SinD(:), uint16([]), uint16([]), uint16([]), int32(options.ndist_side),...
                            %         int32(options.det_w_pseudo), int32(sum(options.pseudot)), int32(options.cryst_per_block), options.nLayers, layer1, layer2);
                        end
                    end
                    if store_coordinates && ascii_ind.world_index1 > 0 && ascii_ind.world_index2 > 0
                        if Nt == 1
                            Rcoordinate = [Rcoordinate, M(:,ascii_ind.world_index1)';M(:,ascii_ind.world_index1+1)';M(:,ascii_ind.world_index1+2)'; M(:,ascii_ind.world_index2)'; M(:,ascii_ind.world_index2+1)'; M(:,ascii_ind.world_index2+2)'];
                        else
                            uind = unique(tIndex);
                            x1 = M(:,ascii_ind.world_index1);
                            x2 = M(:,ascii_ind.world_index2);
                            y1 = M(:,ascii_ind.world_index1 + 1);
                            y2 = M(:,ascii_ind.world_index2 + 1);
                            z1 = M(:,ascii_ind.world_index1 + 2);
                            z2 = M(:,ascii_ind.world_index2 + 2);
                            for uu = 1 : numel(uind)
                                temp = [x1(tIndex == uind(uu))';y1(tIndex == uind(uu))';z1(tIndex == uind(uu))';x2(tIndex == uind(uu))';y2(tIndex == uind(uu))';z2(tIndex == uind(uu))'];
                                Rcoordinate{uu} = [Rcoordinate{uu}, temp];
                            end
                        end
                    end
                end


                if options.verbose
                    disp(['File ' delay_names(lk).name ' loaded'])
                end
            end

        end
        if Nt > 1 && max_time < options.end
            warning(['Specified maximum time ' num2str(options.end) ' is larger than the actual maximum measurement time ' num2str(max_time)])
        end
        clear randoms_index scatter_index apu_randoms L1randoms L2randoms apu_scatter L2scatter L1scatter L2trues L1trues apu_trues trues_index...
            M FOV CC i j k t t1 t2 t3 A S
        if source_index1 ~= 0 && ~isempty(source_index1) && source_index2 ~= 0 && ~isempty(source_index2) && options.source
            save([machine_name '_Ideal_image_' name '_ASCII.mat'],'C')
            if options.store_randoms
                save([machine_name '_Ideal_image_' name '_ASCII.mat'],'RA','-append')
            end
            if options.store_scatter
                save([machine_name '_Ideal_image_' name '_ASCII.mat'],'SC','-append')
            end
        end
        clear Lrandoms Lscatter Ltrues LL1 LL2 C Ldelay2 Ldelay1

    elseif options.use_root

        warning('Loading a single ROOT file can take over 10 minutes depending on the selected options!')
        %% ROOT data
        blocks_per_ring = uint32(blocks_per_ring);
        cryst_per_block = uint32(cryst_per_block);
        det_per_ring = uint32(det_per_ring);
        scatter_components = uint8(options.scatter_components);
        % lisays = uint16(1);
        % large_case = false;
        raw_SinM = raw_SinM(:);
        SinTrues = SinTrues(:);
        SinScatter = SinScatter(:);
        SinRandoms = SinRandoms(:);
        SinD = SinD(:);
        pseudot = options.pseudot;
        if isempty(pseudot)
            pseudot = 0;
        end
        if numel(pseudot) > 1
            pseudot = numel(pseudot);
        end

        % if partitions > 1
        %     save_string_raw = [machine_name '_measurements_' name '_' num2str(partitions) 'timepoints_for_total_of_' num2str(loppu - alku) 's_raw_root.mat'];
        % else
        %     save_string_raw = [machine_name '_measurements_' name '_static_raw_root.mat'];
        % end

        [~,~,ext] = fileparts(fpath);
        if strcmp(ext,'root')
            fnames = dir(fpath);
        else
            % Take coincidence data
            fnames = dir([fpath '/*.root']);
            if size(fnames,1) == 0
                fpath = pwd;
                fpath = [fpath '/'];
                fnames = dir([fpath '*.root']);
                if size(fnames,1) == 0
                    warning('No ROOT (.root) files were found. Input the ROOT output file (if you used a cluster, input the number 1 file).')
                    [file, fpath] = uigetfile({'*.root'},'Select the first GATE ROOT output file');
                    if isequal(file, 0)
                        error('No file was selected!')
                    end
                    fnames = dir([fpath file(1:end-17) '*.root']);
                end
            end
        end

        % if source
        %     if options.obtain_trues
        %         C = cell(7,partitions);
        %     else
        %         C = cell(6,partitions);
        %     end
        %     C(:) = {zeros(options.Nx, options.Ny, options.Nz, 'uint16')};
        %     if options.store_scatter
        %         SC = zeros(options.Nx, options.Ny, options.Nz, 'uint16');
        %     end
        %     if options.store_randoms
        %         RA = zeros(options.Nx, options.Ny, options.Nz, 'uint16');
        %     end
        % end

        %         if options.detectors > 65535
        %             outputUint32 = true;
        % %             lisays = uint32(1);
        %         else
        %             outputUint32 = false;
        %         end

        if options.useIndexBasedReconstruction && options.partitions > 1
            error('Index-based reconstruction is not supported with dynamic data yet!')
        end


        if options.verbose
            disp('Data load started')
        end

        % Go through all the files
        for lk=1:length(fnames)


            nimi = [fpath fnames(lk).name];

            % Use the new features of MATLAB if version is R2019a or newer
            if exist('OCTAVE_VERSION','builtin') == 0 && (verLessThan('matlab','9.6') || options.legacyROOT)
                % [L1, L2, tpoints, A, int_loc, Ltrues, Lscatter, Lrandoms, trues_index, Ldelay1, Ldelay2, int_loc_delay, tpoints_delay, randoms_index, scatter_index, ...
                %     x1, x2, y1, y2, z1, z2, raw_SinM, SinTrues, SinScatter, SinRandoms, SinD] = GATE_root_matlab_C(nimi,vali,alku,loppu, uint32(detectors), blocks_per_ring, ...
                %     cryst_per_block, det_per_ring, uint32(options.linear_multip), source, partitions, options.obtain_trues, options.store_scatter, options.store_randoms, ...
                %     scatter_components, options.randoms_correction, store_coordinates, uint32(cryst_per_block_z), uint32(transaxial_multip), uint32(options.rings), sinoSize, ...
                %     uint32(options.Ndist), uint32(options.Nang), uint32(options.ring_difference), uint32(options.span), uint32(cumsum(options.segment_table)), ...
                %     uint64(options.partitions), sinoSize * uint64(options.TOF_bins), int32(options.ndist_side), options.store_raw_data, raw_SinM, SinTrues, SinScatter, ...
                %     SinRandoms, SinD, int32(options.det_w_pseudo), int32(sum(options.pseudot)), options.TOF_width, FWHM, options.verbose, options.nLayers);
                % tic
                [raw_SinM, SinTrues, SinScatter, SinRandoms, SinD, C, SC, RA, tIndex, coord, Dcoord, trIndex, axIndex, DtrIndex, DaxIndex] = GATE_root_matlab(nimi, partitions, alku, loppu, detectors, blocks_per_ring, cryst_per_block, det_per_ring, options.linear_multip, source, Nt, options.obtain_trues, ...
                    options.store_scatter, options.store_randoms, scatter_components, options.randoms_correction, store_coordinates, transaxial_multip, uint32(cryst_per_block_z), uint32(rings), sinoSize, ...
                    TOF, options.verbose, options.nLayers, options.Ndist, uint32(options.Nang), options.ring_difference, options.span, uint32(cumsum(options.segment_table)), sinoSize(end) * uint64(options.TOF_bins), options.ndist_side, raw_SinM, SinTrues, SinScatter, ...
                    SinRandoms, SinD, uint32(options.det_w_pseudo), pseudot, options.TOF_width, FWHM, dx, dy, dz, bx, by, bz, options.Nx(1), options.Ny(1), options.Nz(1), options.dualLayerSubmodule, options.useIndexBasedReconstruction);
                % toc
            elseif exist('OCTAVE_VERSION','builtin') == 5
                % [L1, L2, tpoints, A, int_loc, Ltrues, Lscatter, Lrandoms, trues_index, Ldelay1, Ldelay2, int_loc_delay, tpoints_delay, randoms_index, scatter_index, ...
                %     x1, x2, y1, y2, z1, z2, raw_SinM, SinTrues, SinScatter, SinRandoms, SinD] = GATE_root_matlab_oct(nimi,vali,alku,loppu, uint32(detectors), blocks_per_ring, ...
                %     cryst_per_block, det_per_ring, uint32(options.linear_multip), source, partitions, options.obtain_trues, options.store_scatter, options.store_randoms, ...
                %     scatter_components, options.randoms_correction, store_coordinates, uint32(cryst_per_block_z), uint32(transaxial_multip), uint32(options.rings), sinoSize, ...
                %     uint32(options.Ndist), uint32(options.Nang), uint32(options.ring_difference), uint32(options.span), uint32(cumsum(options.segment_table)), ...
                %     uint64(options.partitions), sinoSize * uint64(options.TOF_bins), int32(options.ndist_side), options.store_raw_data, raw_SinM, SinTrues, SinScatter, ...
                %     SinRandoms, SinD, int32(options.det_w_pseudo), int32(sum(options.pseudot)), options.TOF_width, FWHM, options.verbose, options.nLayers);
                [raw_SinM, SinTrues, SinScatter, SinRandoms, SinD, C, SC, RA, tIndex, coord, Dcoord, trIndex, axIndex, DtrIndex, DaxIndex] = GATE_root_matlab_oct(nimi, partitions, alku, loppu, detectors, blocks_per_ring, cryst_per_block, det_per_ring, options.linear_multip, source, Nt, options.obtain_trues, ...
                    options.store_scatter, options.store_randoms, scatter_components, options.randoms_correction, store_coordinates, transaxial_multip, uint32(cryst_per_block_z), uint32(rings), sinoSize, ...
                    TOF, options.verbose, options.nLayers, options.Ndist, uint32(options.Nang), options.ring_difference, options.span, uint32(cumsum(options.segment_table)), sinoSize(end) * uint64(options.TOF_bins), options.ndist_side, raw_SinM, SinTrues, SinScatter, ...
                    SinRandoms, SinD, uint32(options.det_w_pseudo), pseudot, options.TOF_width, FWHM, dx, dy, dz, bx, by, bz, options.Nx(1), options.Ny(1), options.Nz(1), options.dualLayerSubmodule, options.useIndexBasedReconstruction);
            else
                % If the machine is large (detectors x detectors matrix is
                % over 2 GB), the data is loaded in a different way
                % if (detectors^2 * 2 >= 1024^3*2 && options.partitions == 1) || ~options.use_raw_data
                %     large_case = true;
                % end
                % [L1, L2, tpoints, A, int_loc, Ltrues, Lscatter, Lrandoms, trues_index, Ldelay1, Ldelay2, int_loc_delay, tpoints_delay, randoms_index, scatter_index, x1, x2, y1, ...
                %     y2, z1, z2, time, timeD, layer1, layer2, layerD1, layerD2] = GATE_root_matlab_MEX(nimi,vali,alku,loppu, detectors, blocks_per_ring, cryst_per_block, det_per_ring, source, partitions, ...
                %     options, scatter_components, store_coordinates, uint32(transaxial_multip), uint32(cryst_per_block_z), uint32(options.rings), large_case, TOF, ...
                %     options.verbose, outputUint32);
                [raw_SinM, SinTrues, SinScatter, SinRandoms, SinD, C, SC, RA, tIndex, coord, Dcoord, trIndex, axIndex, DtrIndex, DaxIndex] = GATE_root_matlab_MEX(nimi, partitions, alku, loppu, detectors, blocks_per_ring, cryst_per_block, det_per_ring, options.linear_multip, source, Nt, options.obtain_trues, ...
                    options.store_scatter, options.store_randoms, scatter_components, options.randoms_correction, store_coordinates, transaxial_multip, uint32(cryst_per_block_z), uint32(rings), sinoSize, ...
                    TOF, options.verbose, options.nLayers, options.Ndist, uint32(options.Nang), options.ring_difference, options.span, uint32(cumsum(options.segment_table)), sinoSize(end) * uint64(options.TOF_bins), options.ndist_side, raw_SinM, SinTrues, SinScatter, ...
                    SinRandoms, SinD, uint32(options.det_w_pseudo), pseudot, options.TOF_width, FWHM, dx, dy, dz, bx, by, bz, options.Nx(1), options.Ny(1), options.Nz(1), options.dualLayerSubmodule, options.useIndexBasedReconstruction);
            end


            % An ideal image can be formed with this. Takes the variables
            % containing the source locations for both singles
            % ll = 1;
            % if source
            %     for jj = int_loc(1) : min(int_loc(2),partitions)
            %         if partitions > 1
            %             S = A(tpoints(ll) + 1:tpoints(ll+1),:);
            %             if options.obtain_trues
            %                 trues_index = logical(Ltrues(tpoints(ll) + 1:tpoints(ll+1),:));
            %             end
            %             if options.store_scatter
            %                 scatter_index = logical(Lscatter(tpoints(ll) + 1:tpoints(ll+1),:));
            %             end
            %             if options.store_randoms
            %                 randoms_index = logical(Lrandoms(tpoints(ll) + 1:tpoints(ll+1),:));
            %             end
            %         else
            %             S = single(A);
            %         end
            %         [C, SC, RA] = formSourceImage(options, S, jj, trues_index, scatter_index, randoms_index, C, SC, RA, tpoints(ll) + 1, tpoints(ll+1));
            %         ll = ll + 1;
            %     end
            % end
            % zerosRemoved = false;
            if partitions == 1
                % if exist('OCTAVE_VERSION','builtin') == 0 && (~verLessThan('matlab','9.6') && ~options.legacyROOT)
                %     if ~options.use_raw_data
                %         ind = L1 == 0;
                %         L1(ind) = [];
                %         if numel(L2) > 1
                %             L2(L2 == 0) = [];
                %         end
                %         zerosRemoved = true;
                %         if outputUint32
                %             ringNumber1 = uint16(idivide(L1 - 1, uint32(det_per_ring)));
                %             ringNumber2 = uint16(idivide(L2 - 1, uint32(det_per_ring)));
                %             ringPos1 = uint16(mod(L1 - 1, uint32(det_per_ring)));
                %             ringPos2 = uint16(mod(L2 - 1, uint32(det_per_ring)));
                %         else
                %             ringNumber1 = idivide(L1 - 1, uint16(det_per_ring));
                %             ringNumber2 = idivide(L2 - 1, uint16(det_per_ring));
                %             ringPos1 = mod(L1 - 1, uint16(det_per_ring));
                %             ringPos2 = mod(L2 - 1, uint16(det_per_ring));
                %         end
                %         if TOF
                %             [bins, discard] = FormTOFBins(options, time(~ind,1));
                %             ringNumber1(discard) = [];
                %             ringNumber2(discard) = [];
                %             ringPos1(discard) = [];
                %             ringPos2(discard) = [];
                %             if options.obtain_trues
                %                 trues_index(discard) = [];
                %             end
                %             if options.store_scatter
                %                 scatter_index(discard) = [];
                %             end
                %             if options.store_randoms
                %                 randoms_index(discard) = [];
                %             end
                %             clear discard
                %         else
                %             bins = 0;
                %             time = alku;
                %         end
                %         [raw_SinM, SinTrues, SinScatter, SinRandoms] = createSinogramASCIICPP(vali, ringPos1, ringPos2, ringNumber1, ringNumber2, trues_index, scatter_index, ...
                %             randoms_index, sinoSize, uint32(options.Ndist), uint32(options.Nang), uint32(options.ring_difference), uint32(options.span), ...
                %             uint32(cumsum(options.segment_table)), sinoSize * uint64(options.TOF_bins), time, uint64(options.partitions), ...
                %             alku, int32(options.det_per_ring), int32(options.rings), uint16(bins), raw_SinM(:), SinTrues(:), SinScatter(:), SinRandoms(:), int32(options.ndist_side), ...
                %             int32(options.det_w_pseudo), int32(sum(options.pseudot)), int32(options.cryst_per_block), options.nLayers, layer1, layer2);
                %         if options.randoms_correction
                %             Ldelay1(Ldelay1 == 0) = [];
                %             Ldelay2(Ldelay2 == 0) = [];
                %             zerosRemoved = true;
                %             if outputUint32
                %                 ringNumber1 = uint16(idivide(Ldelay1 - 1, uint32(det_per_ring)));
                %                 ringNumber2 = uint16(idivide(Ldelay2 - 1, uint32(det_per_ring)));
                %                 ringPos1 = uint16(mod(Ldelay1 - 1, uint32(det_per_ring)));
                %                 ringPos2 = uint16(mod(Ldelay2 - 1, uint32(det_per_ring)));
                %             else
                %                 ringNumber1 = idivide(Ldelay1 - 1, uint16(det_per_ring));
                %                 ringNumber2 = idivide(Ldelay2 - 1, uint16(det_per_ring));
                %                 ringPos1 = mod(Ldelay1 - 1, uint16(det_per_ring));
                %                 ringPos2 = mod(Ldelay2 - 1, uint16(det_per_ring));
                %             end
                %             [SinD, ~, ~, ~] = createSinogramASCIICPP(vali, ringPos1, ringPos2, ringNumber1, ringNumber2, logical([]), logical([]), ...
                %                 logical([]), sinoSize, uint32(options.Ndist), uint32(options.Nang), uint32(options.ring_difference), uint32(options.span), ...
                %                 uint32(cumsum(options.segment_table)), sinoSize, time, uint64(options.partitions), ...
                %                 alku, int32(options.det_per_ring), int32(options.rings), uint16(bins), SinD(:), uint16([]), uint16([]), uint16([]), int32(options.ndist_side), ...
                %                 int32(options.det_w_pseudo), int32(sum(options.pseudot)), int32(options.cryst_per_block), options.nLayers, layerD1, layerD2);
                %         end
                %     end
                % end
                % if options.store_raw_data
                %     % In the "large" case, count matrix needs to be formed like
                %     % in the dynamic case
                %     if large_case
                %         if ~zerosRemoved
                %             L1(L1 == 0) = [];
                %             L2(L2 == 0) = [];
                %         end
                %         L3 = L2;
                %         L2(L2 > L1) = L1(L2 > L1);
                %         L1(L3 > L1) = L3(L3 > L1);
                %         clear L3
                %         if sum(prompts(:)) == 0
                %             prompts = prompts + accumarray([L1 L2],lisays,[detectors detectors],@(x) sum(x,'native'));
                %             if max(prompts(:)) >= 65535
                %                 lisays = uint32(1);
                %                 prompts = accumarray([L1 L2],lisays,[detectors detectors],@(x) sum(x,'native'));
                %             end
                %         else
                %             prompts = prompts + accumarray([L1 L2],lisays,[detectors detectors],@(x) sum(x,'native'));
                %         end
                %         if options.obtain_trues
                %             trues_index = logical(Ltrues);
                %             trues = trues + accumarray([L1(trues_index) L2(trues_index)],lisays,[detectors detectors],@(x) sum(x,'native'));
                %         end
                %         if options.store_scatter
                %             scatter_index = logical(Lscatter);
                %             scatter = scatter + accumarray([L1(scatter_index) L2(scatter_index)],lisays,[detectors detectors],@(x) sum(x,'native'));
                %         end
                %         if options.store_randoms
                %             randoms_index = logical(Lrandoms);
                %             randoms = randoms + accumarray([L1(randoms_index) L2(randoms_index)],lisays,[detectors detectors],@(x) sum(x,'native'));
                %         end
                %         if options.randoms_correction
                %             if ~zerosRemoved
                %                 Ldelay1(Ldelay1 == 0) = [];
                %                 Ldelay2(Ldelay2 == 0) = [];
                %             end
                %             Ldelay3 = Ldelay2;
                %             Ldelay2(Ldelay2 > Ldelay1) = Ldelay1(Ldelay2 > Ldelay1);
                %             Ldelay1(Ldelay3 > Ldelay1) = Ldelay3(Ldelay3 > Ldelay1);
                %             clear Ldelay3
                %             delays = delays + accumarray([Ldelay1 Ldelay2],lisays,[detectors detectors],@(x) sum(x,'native'));
                %         end
                %     else
                %         prompts = prompts + L1;
                %         if options.obtain_trues
                %             trues = trues + Ltrues;
                %         end
                %         if options.store_scatter
                %             scatter = scatter + Lscatter;
                %         end
                %         if options.store_randoms
                %             randoms = randoms + Lrandoms;
                %         end
                %         if options.randoms_correction
                %             delays = delays + Ldelay1;
                %         end
                %     end
                % end
                % Save the interaction coordinates if selected
                if store_coordinates
                    coordinate = [coordinate, coord];
                    if options.randoms_correction
                        Rcoordinate = [Rcoordinate, Dcoord];
                    end
                    % temp = [x1, x2];
                    % x_coordinate = [x_coordinate; temp];
                    % temp = [y1, y2];
                    % y_coordinate = [y_coordinate; temp];
                    % temp = [z1, z2];
                    % z_coordinate = [z_coordinate; temp];
                end
                if options.useIndexBasedReconstruction
                    trIndices = [trIndices,trIndex];
                    axIndices = [axIndices,axIndex];
                    if options.randoms_correction
                        DtrIndices = [DtrIndices,DtrIndex];
                        DaxIndices = [DaxIndices,DaxIndex];
                    end
                end
            else
                % if int_loc(1) > 0
                % ll = 1;
                % if exist('OCTAVE_VERSION','builtin') == 0 && (~verLessThan('matlab','9.6') && ~options.legacyROOT)
                %     if ~options.use_raw_data
                %         if outputUint32
                %             ringNumber1 = uint16(idivide(L1 - 1, uint32(det_per_ring)));
                %             ringNumber2 = uint16(idivide(L2 - 1, uint32(det_per_ring)));
                %             ringPos1 = uint16(mod(L1 - 1, uint32(det_per_ring)));
                %             ringPos2 = uint16(mod(L2 - 1, uint32(det_per_ring)));
                %         else
                %             ringNumber1 = idivide(L1 - 1, uint16(det_per_ring));
                %             ringNumber2 = idivide(L2 - 1, uint16(det_per_ring));
                %             ringPos1 = mod(L1 - 1, uint16(det_per_ring));
                %             ringPos2 = mod(L2 - 1, uint16(det_per_ring));
                %         end
                %         ind = time(:,1) <= 0;
                %         ringNumber1(ind) = [];
                %         ringNumber2(ind) = [];
                %         ringPos1(ind) = [];
                %         ringPos2(ind) = [];
                %         time(ind,:) = [];
                %         if TOF
                %             [bins, discard] = FormTOFBins(options, time);
                %             ringNumber1(discard) = [];
                %             ringNumber2(discard) = [];
                %             ringPos1(discard) = [];
                %             ringPos2(discard) = [];
                %             time(discard,:) = [];
                %             if options.obtain_trues
                %                 trues_index(discard) = [];
                %             end
                %             if options.store_scatter
                %                 scatter_index(discard) = [];
                %             end
                %             if options.store_randoms
                %                 randoms_index(discard) = [];
                %             end
                %             clear discard
                %         else
                %             bins = 0;
                %         end
                %         [raw_SinM, SinTrues, SinScatter, SinRandoms] = createSinogramASCIICPP(vali, ringPos1, ringPos2, ringNumber1, ringNumber2, trues_index, scatter_index, ...
                %             randoms_index, sinoSize, uint32(options.Ndist), uint32(options.Nang), uint32(options.ring_difference), uint32(options.span), ...
                %             uint32(cumsum(options.segment_table)), sinoSize * uint64(options.TOF_bins), time(:,1), uint64(options.partitions), ...
                %             alku, int32(options.det_w_pseudo), int32(options.rings), uint16(bins), raw_SinM(:), SinTrues(:), SinScatter(:), SinRandoms(:), int32(options.ndist_side), ...
                %             int32(options.det_w_pseudo), int32(sum(options.pseudot)), int32(options.cryst_per_block));
                %         if options.randoms_correction
                %             if outputUint32
                %                 ringNumber1 = uint16(idivide(Ldelay1 - 1, uint32(det_per_ring)));
                %                 ringNumber2 = uint16(idivide(Ldelay2 - 1, uint32(det_per_ring)));
                %                 ringPos1 = uint16(mod(Ldelay1 - 1, uint32(det_per_ring)));
                %                 ringPos2 = uint16(mod(Ldelay2 - 1, uint32(det_per_ring)));
                %             else
                %                 ringNumber1 = idivide(Ldelay1 - 1, uint16(det_per_ring));
                %                 ringNumber2 = idivide(Ldelay2 - 1, uint16(det_per_ring));
                %                 ringPos1 = mod(Ldelay1 - 1, uint16(det_per_ring));
                %                 ringPos2 = mod(Ldelay2 - 1, uint16(det_per_ring));
                %             end
                %             ind = timeD <= 0;
                %             ringNumber1(ind) = [];
                %             ringNumber2(ind) = [];
                %             ringPos1(ind) = [];
                %             ringPos2(ind) = [];
                %             timeD(ind) = [];
                %             [SinD, ~, ~, ~] = createSinogramASCIICPP(vali, ringPos1, ringPos2, ringNumber1, ringNumber2, logical([]), logical([]), ...
                %                 logical([]), sinoSize, uint32(options.Ndist), uint32(options.Nang), uint32(options.ring_difference), uint32(options.span), ...
                %                 uint32(cumsum(options.segment_table)), sinoSize, timeD, uint64(options.partitions), ...
                %                 alku, int32(options.det_w_pseudo), int32(options.rings), uint16(bins), SinD(:), uint16([]), uint16([]), uint16([]), int32(options.ndist_side), ...
                %                 int32(options.det_w_pseudo), int32(sum(options.pseudot)), int32(options.cryst_per_block));
                %         end
                %     end
                % end
                % for kk = int_loc(1) : min(int_loc(2),partitions)
                % if options.store_raw_data
                %     % Dynamic case
                %     apu1 = L1(tpoints(ll) + 1:tpoints(ll+1));
                %     apu2 = L2(tpoints(ll) + 1:tpoints(ll+1));
                %     if exist('OCTAVE_VERSION','builtin') == 0 && (~verLessThan('matlab','9.6') && ~options.legacyROOT)
                %         L3 = apu2;
                %         apu2(apu2 > apu1) = apu1(apu2 > apu1);
                %         apu1(L3 > apu1) = L3(L3 > apu1);
                %         clear L3
                %     end
                %     inde = any(apu1,2);
                %     apu1 = apu1(inde,:);
                %     apu2 = apu2(inde,:);
                %     if isempty(prompts{kk})
                %         prompts{kk} = accumarray([apu1 apu2],1,[detectors detectors],[],[],true);
                %     else
                %         prompts{kk} = prompts{kk} + accumarray([apu1 apu2],1,[detectors detectors],[],[],true);
                %     end
                %     if options.obtain_trues
                %         trues_index = logical(Ltrues(tpoints(ll) + 1:tpoints(ll+1)));
                %         trues_index = trues_index(inde);
                %         if isempty(trues{kk})
                %             trues{kk} = accumarray([apu1(trues_index) apu2(trues_index)],1,[detectors detectors],[],[],true);
                %         else
                %             trues{kk} = trues{kk} + accumarray([apu1(trues_index) apu2(trues_index)],1,[detectors detectors],[],[],true);
                %         end
                %     end
                %     if options.store_scatter
                %         trues_index = logical(Lscatter(tpoints(ll) + 1:tpoints(ll+1)));
                %         trues_index = trues_index(inde);
                %         if isempty(scatter{kk})
                %             scatter{kk} = accumarray([apu1(trues_index) apu2(trues_index)],1,[detectors detectors],[],[],true);
                %         else
                %             scatter{kk} = scatter{kk} + accumarray([apu1(trues_index) apu2(trues_index)],1,[detectors detectors],[],[],true);
                %         end
                %     end
                %     if options.store_randoms
                %         trues_index = logical(Lrandoms(tpoints(ll) + 1:tpoints(ll+1)));
                %         trues_index = trues_index(inde);
                %         if isempty(randoms{kk})
                %             randoms{kk} = accumarray([apu1(trues_index) apu2(trues_index)],1,[detectors detectors],[],[],true);
                %         else
                %             randoms{kk} = randoms{kk} + accumarray([apu1(trues_index) apu2(trues_index)],1,[detectors detectors],[],[],true);
                %         end
                %     end
                % end
                if store_coordinates
                    uind = unique(tIndex);
                    for uu = 1 : numel(uind)
                        coordinate{uu} = [coordinate{uu}, coord(:,tIndex == uind(uu))];
                    end
                    if options.randoms_correction
                        warning('Coordinates for delayed coincidences are not supported for list-mode data in dynamic mode!')
                    end
                end
                % ll = ll + 1;
                % end
                % if options.randoms_correction && options.store_raw_data
                %     if  int_loc_delay(1) > 0
                %         ll = 1;
                %         for kk = int_loc_delay(1) : min(int_loc_delay(2),partitions)
                %             apu1 = Ldelay1(tpoints_delay(ll) + 1:tpoints_delay(ll+1));
                %             apu2 = Ldelay2(tpoints_delay(ll) + 1:tpoints_delay(ll+1));
                %             if exist('OCTAVE_VERSION','builtin') == 0 && (~verLessThan('matlab','9.6') && ~options.legacyROOT)
                %                 L3 = apu2;
                %                 apu2(apu2 > apu1) = apu1(apu2 > apu1);
                %                 apu1(L3 > apu1) = L3(L3 > apu1);
                %                 clear L3
                %             end
                %             inde = any(apu1,2);
                %             apu1 = apu1(inde,:);
                %             apu2 = apu2(inde,:);
                %             if isempty(delays{kk})
                %                 delays{kk} = accumarray([apu1 apu2],1,[detectors detectors],[],[],true);
                %             else
                %                 delays{kk} = delays{kk} + accumarray([apu1 apu2],1,[detectors detectors],[],[],true);
                %             end
                %             ll = ll + 1;
                %         end
                %     end
                % end
                % end
            end
            if options.verbose
                disp(['File ' fnames(lk).name ' loaded'])
            end

        end
        clear Lrandoms apu2 Lscatter Ltrues apu1 trues_index FOV CC i j k t t1 t2 t3 A S inde
        if source
            save([machine_name '_Ideal_image_coordinates_' name '_ROOT.mat'],'C')
            if options.store_randoms
                save([machine_name '_Ideal_image_coordinates_' name '_ROOT.mat'],'RA','-append')
            end
            if options.store_scatter
                save([machine_name '_Ideal_image_coordinates_' name '_ROOT.mat'],'SC','-append')
            end
        end
        clear C RA SC GATE_root_matlab_MEX.m


    end
end
%% Form the raw data vector and the sinograms

% coincidences = cell(partitions,1);
% if options.use_machine == 0 && options.obtain_trues
%     true_coincidences = cell(partitions,1);
% end
% if options.use_machine == 0 && options.store_scatter
%     scattered_coincidences = cell(partitions,1);
% end
% if options.use_machine == 0 && options.store_randoms
%     random_coincidences = cell(partitions,1);
% end
% if options.randoms_correction && (~options.use_LMF || options.use_machine == 1)
%     delayed_coincidences = cell(partitions,1);
% end
%
% tot_counts = 0;
% if options.use_machine == 0 && options.obtain_trues
%     tot_trues = 0;
% end
% if options.use_machine == 0 && options.store_scatter
%     tot_scatter = 0;
% end
% if options.use_machine == 0 && options.store_randoms
%     tot_randoms = 0;
% end
% if options.randoms_correction && (~options.use_LMF || options.use_machine == 1)
%     tot_delayed = 0;
% end

if ~options.use_raw_data && ~store_coordinates && ~options.useIndexBasedReconstruction && ~isempty(raw_SinM)
    if size(raw_SinM,1) == numel(raw_SinM)
        raw_SinM = reshape(raw_SinM, options.Ndist, options.Nang(end), totSinos, options.nLayers^2, options.TOF_bins, Nt);
        if options.obtain_trues
            SinTrues = reshape(SinTrues, options.Ndist, options.Nang, totSinos, options.nLayers^2, options.TOF_bins, Nt);
        end
        if options.store_scatter
            SinScatter = reshape(SinScatter, options.Ndist, options.Nang, totSinos, options.nLayers^2, options.TOF_bins, Nt);
        end
        if options.store_randoms
            SinRandoms = reshape(SinRandoms, options.Ndist, options.Nang, totSinos, options.nLayers^2, options.TOF_bins, Nt);
        end
    end
    if options.randoms_correction && size(SinD,1) == numel(SinD)
        SinD = reshape(SinD, options.Ndist, options.Nang, totSinos, options.nLayers^2, Nt);
    end
    if Nt > 1
        raw_SinM = squeeze(num2cell(raw_SinM, (1 : ndims(raw_SinM) - 1)));
        if options.obtain_trues
            SinTrues = squeeze(num2cell(SinTrues, (1 : ndims(SinTrues) - 1)));
        end
        if options.store_scatter
            SinScatter = squeeze(num2cell(SinScatter, (1 : ndims(SinScatter) - 1)));
        end
        if options.store_randoms
            SinRandoms = squeeze(num2cell(SinRandoms, (1 : ndims(SinRandoms) - 1)));
        end
        if options.randoms_correction
            SinD = squeeze(num2cell(SinD, [1 2 3]));
        end
    end
    if ~options.randoms_correction
        SinD = [];
    end
    if ~options.obtain_trues
        SinTrues = [];
    end
    if ~options.store_scatter
        SinScatter = [];
    end
    if ~options.store_randoms
        SinRandoms = [];
    end
    form_sinograms(options, true, raw_SinM, SinTrues, SinScatter, SinRandoms, SinD);
end

if options.store_raw_data
    % for llo=1:partitions
    %
    %     % Take only the lower triangular part and store as a sparse vector
    %     if partitions > 1
    %         coincidences{llo, 1} = sparse(double(prompts{llo}(tril(true(size(prompts{llo})), 0))));
    %     else
    %         coincidences{llo, 1} = sparse(double(prompts(tril(true(size(prompts)), 0))));
    %     end
    %
    %     if options.verbose
    %         if partitions > 1
    %             counts = sum(prompts{llo}(:));
    %         else
    %             counts = sum(prompts(:));
    %         end
    %     end
    %
    %     if options.verbose
    %         disp(['Total number of prompts at time point ' num2str(llo) ' is ' num2str(counts) '.'])
    %         tot_counts = tot_counts + counts;
    %     end
    %
    %     if options.use_machine == 0 && options.obtain_trues
    %         if partitions > 1
    %             true_coincidences{llo, 1} = sparse(double(trues{llo}(tril(true(size(trues{llo})), 0))));
    %         else
    %             true_coincidences{llo, 1} = sparse(double(trues(tril(true(size(trues)), 0))));
    %         end
    %         if options.verbose
    %             if partitions > 1
    %                 Tcounts = sum(trues{llo}(:));
    %             else
    %                 Tcounts = sum(trues(:));
    %             end
    %             disp(['Total number of trues at time point ' num2str(llo) ' is ' num2str(Tcounts) '.'])
    %             tot_trues = tot_trues + Tcounts;
    %         end
    %     end
    %     if options.use_machine == 0 && options.store_scatter
    %         if partitions > 1
    %             scattered_coincidences{llo, 1} = sparse(double(scatter{llo}(tril(true(size(scatter{llo})), 0))));
    %         else
    %             scattered_coincidences{llo, 1} = sparse(double(scatter(tril(true(size(scatter)), 0))));
    %         end
    %         if options.verbose
    %             if partitions > 1
    %                 Scounts = sum(scatter{llo}(:));
    %             else
    %                 Scounts = sum(scatter(:));
    %             end
    %             disp(['Total number of scattered coincidences at time point ' num2str(llo) ' is ' num2str(Scounts) '.'])
    %             tot_scatter = tot_scatter + Scounts;
    %         end
    %     end
    %     if options.use_machine == 0 && options.store_randoms
    %         if partitions > 1
    %             random_coincidences{llo, 1} = sparse(double(randoms{llo}(tril(true(size(randoms{llo})), 0))));
    %         else
    %             random_coincidences{llo, 1} = sparse(double(randoms(tril(true(size(randoms)), 0))));
    %         end
    %         if options.verbose
    %             if partitions > 1
    %                 Rcounts = sum(randoms{llo}(:));
    %             else
    %                 Rcounts = sum(randoms(:));
    %             end
    %             disp(['Total number of randoms at time point ' num2str(llo) ' is ' num2str(Rcounts) '.'])
    %             tot_randoms = tot_randoms + Rcounts;
    %         end
    %     end
    %     if options.randoms_correction && (~options.use_LMF || options.use_machine == 1)
    %         if partitions > 1
    %             delayed_coincidences{llo, 1} = sparse(double(delays{llo}(tril(true(size(delays{llo})), 0))));
    %         else
    %             delayed_coincidences{llo, 1} = sparse(double(delays(tril(true(size(delays)), 0))));
    %         end
    %         if options.verbose
    %             if partitions > 1
    %                 Dcounts = sum(delays{llo}(:));
    %             else
    %                 Dcounts = sum(delays(:));
    %             end
    %             disp(['Total number of delayed coincidences at time point ' num2str(llo) ' is ' num2str(Dcounts) '.'])
    %             tot_delayed = tot_delayed + Dcounts;
    %         end
    %     end
    % end
    % if options.verbose
    %     disp('Saving data...')
    % end
    % if partitions == 1
    %     if exist('OCTAVE_VERSION', 'builtin') == 0
    %         save(save_string_raw, 'coincidences', '-v7.3')
    %     else
    %         save(save_string_raw, 'coincidences','-v7')
    %     end
    %     for variableIndex = 1:length(variableList)
    %         if exist(variableList{variableIndex},'var')
    %             if exist('OCTAVE_VERSION', 'builtin') == 0
    %                 save(save_string_raw,variableList{variableIndex},'-append')
    %             else
    %                 save(save_string_raw,variableList{variableIndex},'-append','-v7')
    %             end
    %         end
    %     end
    % else
    %     if options.verbose
    %         disp(['Total number of prompts at all time points is ' num2str(tot_counts) '.'])
    %         if options.obtain_trues && options.use_machine == 0
    %             disp(['Total number of trues at all time points is ' num2str(tot_trues) '.'])
    %         end
    %         if options.store_scatter && options.use_machine == 0
    %             disp(['Total number of scattered coincidences at all time points is ' num2str(tot_scatter) '.'])
    %         end
    %         if options.store_randoms && options.use_machine == 0
    %             disp(['Total number of randoms at all time points is ' num2str(tot_randoms) '.'])
    %         end
    %         if options.randoms_correction && (~options.use_LMF || options.use_machine == 1)
    %             disp(['Total number of delayed coincidences at all time points is ' num2str(tot_delayed) '.'])
    %         end
    %     end
    %     if exist('OCTAVE_VERSION', 'builtin') == 0
    %         save(save_string_raw, 'coincidences', '-v7.3')
    %     else
    %         save(save_string_raw, 'coincidences','-v7')
    %     end
    %     for variableIndex = 1:length(variableList)
    %         if exist(variableList{variableIndex},'var')
    %             if exist('OCTAVE_VERSION', 'builtin') == 0
    %                 save(save_string_raw,variableList{variableIndex},'-append')
    %             else
    %                 save(save_string_raw,variableList{variableIndex},'-append','-v7')
    %             end
    %         end
    %     end
    % end
    % if nargout >= 1
    %     varargout{1} = coincidences;
    % end
    % if nargout >= 2 && exist('delayed_coincidences','var')
    %     varargout{2} = delayed_coincidences;
    % end
    % if nargout >= 3 && exist('true_coincidences','var')
    %     varargout{3} = true_coincidences;
    % end
    % if nargout >= 4 && exist('scattered_coincidences','var')
    %     varargout{4} = scattered_coincidences;
    % end
    % if nargout >= 5 && exist('random_coincidences','var')
    %     varargout{5} = random_coincidences;
    % end
else
    if nargout >= 1
        varargout{1} = raw_SinM;
    end
    if nargout >= 2 && exist('SinD','var')
        varargout{2} = SinD;
    end
    if nargout >= 3 && exist('SinTrues','var')
        varargout{3} = SinTrues;
    end
    if nargout >= 4 && exist('SinScatter','var')
        varargout{4} = SinScatter;
    end
    if nargout >= 5 && exist('SinRandoms','var')
        % varargout{5} = SinRandoms;
    end
end
if nargout >= 6
    if options.useIndexBasedReconstruction
        varargout{6} = trIndices;
        varargout{7} = axIndices;
        if options.randoms_correction && nargout >= 9
            varargout{8} = DtrIndices;
            varargout{9} = DaxIndices;
        elseif nargout >= 8
            warning('Delayed coincidence coordinates selected as output but no randoms correction selected. No indices will be output!')
            varargout{8} = [];
            if nargout >= 9
                varargout{9} = [];
            end
        end
    else
        varargout{6} = coordinate;
        if options.randoms_correction && nargout >= 7
            varargout{7} = Rcoordinate;
        elseif nargout >= 7
            warning('Delayed coincidence coordinates selected as output but no randoms correction selected. No coordinates will be output!')
            varargout{7} = [];
        end
    end
    % varargout{7} = y_coordinate;
    % varargout{8} = z_coordinate;
end
if options.verbose
    disp('Measurements loaded and saved')
    toc
end
end