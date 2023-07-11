% This code is written by Marcus Albrechtsen 19-03-2020.
% Contact on maralb@fotonik.dtu.dk or marcus.albrechtsen@gmail.com
% No rights reserved.

clear all;
warning('off', 'MATLAB:polyshape:repairedBySimplify');

pathstr = 'edc-si2/OSC_ND100x4nm_1540nmHCW_Si_WB';
out_fname = 'EDC-Si2-outline';

curveID = 1;
curvename = @(id) sprintf('%s/DesignSymCurves_%d.txt',pathstr,id);

plot_figure_number = 1; % if set to zero no plot is generated.



%% GDS units: GDS write polygon as int32 (signed), i.e. +/- 2.14e9 is max.
% That is, the curves must be scaled and rounded to int32, and it must be
% specified what the min increment corresponds to in meters (1e-9 for nm).

% The unit of the software reading the GDS must also be specified. This is
% specified in terms of database-units; i.e. the precision of the rounding.
% If nm (1e-9) is the rounding precision and um is the working unit then
% this second number should be (1e-3). If pm precision is used to desribe
% the boundaries, and nm is the working unit, this second number is still
% 1e-3. If pm and um are desired then it must be 1e-6, etc.

curve_input_xy_unit = 1e-9; % specify unit of input curves in meters.

gdsUNITS.databaseToMeter = 1e-13; % Specify value of bit-increment.
gdsUNITS.databaseToUser = 1e-4; % Specify scaling of polygons/user unit.
gdsUNITS.userUnits = gdsUNITS.databaseToMeter/gdsUNITS.databaseToUser; % m.
gdsUNITS.maxValueInMeter = gdsUNITS.databaseToMeter*2^31; % unit m; can be +/-.

%% Load all curve-names in specified folder into a ps_root.
% NB. union force polygon to not be closed unless 'simplified' is set to 'false'.
% The gds_writer always adds first vertex to end of boundary to close boundaries.

ps_root = polyshape();
polygon_scale = curve_input_xy_unit / gdsUNITS.userUnits;
while exist(curvename(curveID),'file')
    bndry = load(curvename(curveID)) * polygon_scale;
    ps_root = union(ps_root, polyshape(bndry(:,2)-1,bndry(:,1)-1) );
    curveID = curveID + 1;
end


%% Perform some manipulations to the design: recenter and extract c-hull.

% First we recenter th design around origo; eases symmetries in COMSOL.
[bbx,bby] = boundingbox(ps_root); w = bbx(2)-bbx(1); h = bby(2)-bby(1);
ps_root = translate(ps_root, -w/2-bbx(1),-h/2-bby(1));

if max([w h]) * gdsUNITS.userUnits > gdsUNITS.maxValueInMeter
    warning('FATAL ERROR! Geometry outside GDS II. Reduce precision or scale.');
    return;
end

% Extract the convex hull of the design. Use this to subtract the main PS
% from in any CAD program (intersect). Write this to a different layer.
domain_boundary = convhull(ps_root);

% Do we want to plot it in MATLAB, figure: "plot_figure_number"?
if plot_figure_number
    fh = figure(plot_figure_number); clf(fh); fh.WindowStyle = 'docked'; hold on;
    h2 = plot(domain_boundary); h2.FaceAlpha = 1/7; plot(ps_root); box on; hold off;
    xlabel('Position, {\itx} (nm)'); ylabel('Position, {\ity} (nm)'); set(gca,'FontSize',20);
end


%% Write polyshape to GDS.
fid = open_gds([pathstr '/' out_fname '.gds'],gdsUNITS);

% Step 1: Define layer/datatype 0 to contain entire geometry boundary,
% which can be used in COMSOL to invert design (using "intersect").
    export_ps_outline_to_GDS(fid,domain_boundary,0,gdsUNITS);

% Step 2: Define the outlines of each feature, if the design is already
% well behaved, i.e. only concave, non-intersecting polygons, this will not
% change the input, and "features_holes" will be empty.
feature_boundaries = rmholes(ps_root);
features_holes = holes(ps_root);
    export_ps_outline_to_GDS(fid,feature_boundaries,1,gdsUNITS);
    
if length(features_holes), export_ps_outline_to_GDS(fid,features_holes,2,gdsUNITS); end

close_gds_file(fid); % Write footer, flush fwrite and close fid.




%% Worker functions to export a GDS file
% Brief GDS II doc: https://www.iue.tuwien.ac.at/phd/minixhofer/node52.html
% Full doc: http://www.bitsavers.org/pdf/calma/GDS_II_Stream_Format_Manual_6.0_Feb87.pdf

function export_ps_outline_to_GDS(fid,polyshape_obj,layer,gdsUNITS)
% Bulk: write polygon outlines: NB! Polygons must be closed, cf. standard.
    geom_scale = 1 / gdsUNITS.databaseToUser;
    datatype = layer;
    pVec = regions(polyshape_obj);
    for j=1:polyshape_obj.NumRegions
        fwrite(fid,[4,2048],'uint16'); % Boundary
        fwrite(fid,[6,3330],'uint16'); % Layer
        fwrite(fid,layer,'uint16');
        fwrite(fid,[6,3586,datatype],'uint16');
    
        vert = int32(geom_scale * pVec(j).Vertices([1:end 1],:)).'; % We write with pm precision and call unit nm
        fwrite(fid,[8*length(vert) + 4,4099],'uint16'); % 32 bit gives 2; xy gives 2;
        fwrite(fid,vert(:),'int32');
        
        fwrite(fid,[4,4352],'uint16'); % End
    end
end

function close_gds_file(fid)
    fwrite(fid,[4,1792],'uint16'); fwrite(fid,[4,1024],'uint16');
    pos=ftell(fid); for i=1:(2048-mod(pos,2048)), fwrite(fid,0,'uint8'); end
    fclose(fid);
end

function fid = open_gds(filename,gdsUNITS)
    fid=fopen(filename,'w','b');

% Write header and BGNLIB
    fwrite(fid,typecast(uint16([6 2 3]),'uint16'),'uint16');
    [Y, M, D, H, MN, S] = datevec(now);
    fwrite(fid,[28, 258, Y, M, D, H, MN, uint16(S), Y, M, D, H, MN, uint16(S)],'uint16');

% Libname
    L = length(filename);
    fwrite(fid,[L+mod(L,2)+4,518],'uint16');
    fwrite(fid,filename,'char*1');
    if mod(L,2), fwrite(fid,0,'uint8'); end
    
% UNITS: This is two 8-byte floats, which must match the polygon-scaling.
    % Allocate: 20 (hex:0014) bytes; first 4 is type: 773 (hex:0305).
    % Meaning of 0305: '03' is UNITS record type; '05' is 8-byte real type.
    fwrite(fid,typecast(uint16([hex2dec('0014') hex2dec('0305')]),'uint16'),'uint16'); % [20 773]
    %fwrite(fid,typecast(swapbytes(hex2num('3e4189374bc6a7f0')),'uint8'),'uint8'); % 1e-3
    %fwrite(fid,typecast(swapbytes(hex2num('3944b82fa09b5a54')),'uint8'),'uint8'); % 1e-9
    
    % First input is database-unit in user units; gdsUNITS.databaseToUser.
    % Second input is database-unit in meters: gdsUNITS.databaseToMeter.
    % Format: SEEE EEEE MMMM MMMM MMMM MMMM: S sign, E exponent, M mantissa
    % However, sign must be 0 here, or can be used in scale?
    % Value: (-1)^s * mantisa * 16^(64-exponent). Encoding is hexadecimal.
    hexStr1 = dec2float8equivalent_hexStr(gdsUNITS.databaseToMeter);
    hexStr2 = dec2float8equivalent_hexStr(gdsUNITS.databaseToUser);

    fwrite(fid,typecast(swapbytes(hex2num(hexStr2)),'uint8'),'uint8');
    fwrite(fid,typecast(swapbytes(hex2num(hexStr1)),'uint8'),'uint8');


% BGNSTR and strcutre name; call "Layout".
    fwrite(fid,[28, 1282, Y, M, D, H, MN, uint16(S), Y, M, D, H, MN, uint16(S)],'uint16');
    fwrite(fid,[10,1542],'uint16'); fwrite(fid,'Layout','char*1');
end

function hexStr = dec2float8equivalent_hexStr(target)
    significand_bits = 56;

    val_m = 2.^(-1:-1:-significand_bits);
    exponent = - (find(val_m(4:4:end) <= target, 1)-1); % Offset is in hex, i.e. base 16 or 2^4.

    rem = target * 16^-exponent;
    significand = zeros(1,significand_bits);
    for idx = 1:significand_bits
        if val_m(idx) <= rem
            rem = rem - val_m(idx);
            significand(idx) = 1;
        end
        if rem == 0, break; end
    end

    significand = reshape(significand, [4,14]);
    significand = dec2hex(bi2de(significand(end:-1:1,:).')).';
    hexStr = [dec2hex(exponent+64) significand];
end