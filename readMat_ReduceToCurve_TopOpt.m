%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Converts density field to closed curve segments according to a number of 
% user parameters utilizing a number of costom functions and the 
% isocontour.m function by D.Kroon University of Twente (March 2011).
%
% 
% Written by: REC@MEK.DTU.DK in year 2018
%
% Disclaimer:                                                              
% The author reserves all rights but does not guaranty that the code is   
% free from errors. Furthermore, we shall not be liable in any event     
% caused by the use of the program.
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% close all
clear all

% IMPLEMENTING MULTI REGION FILTERING

% Loading image or matrix
DATA_TYPE = 'MATRIX'; % Options: IMAGE, MATRIX
PathToData = '/home/cme-ex1bmdaj/Desktop/Projects/TOPS/WAVEGUIDE_EzHz_heat_coupled_minmax_nedelec_Q8/optimized.mat';
DataName = 'matrix';
OutPutPath = '/home/cme-ex1bmdaj/Desktop/Projects/TOPS/WAVEGUIDE_EzHz_heat_coupled_minmax_nedelec_Q8/';
OutPutName = 'design_curves_3D_wg';

% USER PARAMETERS
% USER PARAMETERS
SymmetryEnforcementX = 0;
SymmetryEnforcementY = 0;
ModelReduction.critical_angle = 0.1 * pi/180; % Maximum angle between vectors created from geometry points
ModelReduction.number_of_vertices_jumped_in_check = 3; % Number of points between geometry points
ModelReduction.max_vertex_distance = 5; % Maximum distance between two points in reduced geometry
ModelReduction.min_vertex_distance = 1; % Minimum distance between two points in reduced geometry

% User specify final image size (in arbirary area unit)
sizeX_Original = 1;
sizeX_Output = 0.05*1000;
size_Scaling = sizeX_Output/sizeX_Original;

% User choice of projection level (contour level for curve)
ProjectionLevel = 0.5; %0.5; 

% User choice of additional filtering
FILTER_AGAIN = 0;
filter_range = 0;
FilterPaddingType = 0; % zeros/ones (solid/void)
% USER PARAMETERS
% USER PARAMETERS

% Input data
if strcmp(DATA_TYPE,'IMAGE')
    % Load image
    Data = double(imread(PathToData));
    % Rescale image data to [0,1];
    Data = Data/max(Data(:));
    % Convert from rgb to grayscale
    Data = rgb2gray(Data);
elseif strcmp(DATA_TYPE,'MATRIX')
    Data = load(PathToData);
    Data = Data.([DataName]);
    Data(isnan(Data)) = 0;
else
    error('The input data type is not viable, use IMAGE or MATRIX');
end


% Filter regions
nFilterRegions = 1;
regionDesignations = zeros(nFilterRegions,2,2);
regionDesignations(1,1,:) = [0.48,0.52];
regionDesignations(1,2,:) = [0.48,0.52];
Bpadded_struct = struct;

% Output
OutputFileName = [OutPutName '_ReFilter_' num2str(max(filter_range(:))) '_PL_' num2str(ProjectionLevel)];
OutputFileNameSmoothedDesign = [OutPutName '_ReFilter_' num2str(max(filter_range(:))) '_UnitCell'];

% User specified maxinum number of vertices per curve
max_number_of_vertices_per_curve = 6000;

% Read matrix
A = Data;
if SymmetryEnforcementX==1
A(:,1:end/2) = A(:,end:-1:end/2+1);
end
if SymmetryEnforcementY==1
A(1:end/2,:) = A(end:-1:end/2+1,:);
end

% Extract desired data [MANUAL OPERATIONS!!!]
B = A;
% B(B<0.5) = 0;

% 0-padding image to avoid open curves at edges of image
PaddingSize = 2*filter_range;
Bpadded = FilterPaddingType * ones(size(B)+PaddingSize);
Bpadded(1+PaddingSize/2:end-PaddingSize/2,1+PaddingSize/2:end-PaddingSize/2) = B;

% Scaling data to [0,1]
Bpadded = 1.0-Bpadded/max(Bpadded(:));

% Size of matrix
[numberElementsY,numberElementsX]=size(Bpadded);

% Apply filtering if desired
if FILTER_AGAIN==1
    [dy,dx] = meshgrid(-ceil(filter_range)+1:ceil(filter_range)-1,-ceil(filter_range)+1:ceil(filter_range)-1);
    h = max(0,filter_range-sqrt(dx.^2+dy.^2));
    Hs_D = conv2(ones(numberElementsY,numberElementsX),h,'same');
    
    % filter the matrix
    Bpadded = conv2((Bpadded .* ones(numberElementsY,numberElementsX))./Hs_D,h,'same');
end

BpaddedRestricted = Bpadded(1+PaddingSize/2:end-PaddingSize/2,1+PaddingSize/2:end-PaddingSize/2);

BpaddedRestrictedAddingBlankBoarder = (1-FilterPaddingType)*ones(size(BpaddedRestricted)+2);
BpaddedRestrictedAddingBlankBoarder(2:end-1,2:end-1) = BpaddedRestricted;

UnitCell_PostFilter = BpaddedRestrictedAddingBlankBoarder;
save([OutputFileNameSmoothedDesign '.mat'],'UnitCell_PostFilter','ProjectionLevel','ModelReduction');

% MS algo
[Lines,Vertices,Objects]=isocontour(BpaddedRestrictedAddingBlankBoarder,ProjectionLevel);

% Identify the number of curves and associated lines and vertices
[ curve_index_list, number_of_curves ] = IdentifyClosedCurves( Lines );

% Sort vertices to single connected curve.
[ all_closed_curves ] = FromVLRepresentationToClosedCurve( Vertices, Lines, curve_index_list );

% Reduce curve resolution to reduce number of points constituting final geometry
[ all_closed_curves ] = ReduceCurveResolution( number_of_curves, all_closed_curves, ModelReduction );

% Rescaling geometry size
for idx_curve = 1:number_of_curves
    all_closed_curves.(['Curve_' num2str(idx_curve)])=all_closed_curves.(['Curve_' num2str(idx_curve)])*size_Scaling;
    all_closed_curves.(['Reduced_Curve_Min_Only_' num2str(idx_curve)])=all_closed_curves.(['Reduced_Curve_Min_Only_' num2str(idx_curve)])*size_Scaling;
    all_closed_curves.(['Reduced_Curve_All_' num2str(idx_curve)])=all_closed_curves.(['Reduced_Curve_All_' num2str(idx_curve)])*size_Scaling;
end

% Plot result
figure(30);
for idx_curve = 1:number_of_curves
    vertex_closed_curve = all_closed_curves.(['Curve_' num2str(idx_curve)]);
    vertex_closed_curve_reduced = all_closed_curves.(['Reduced_Curve_All_' num2str(idx_curve)]);
    vertex_closed_curve_min_only_reduced = all_closed_curves.(['Reduced_Curve_Min_Only_' num2str(idx_curve)]);
    plot(vertex_closed_curve(:,2),vertex_closed_curve(:,1),'-b*');
    hold on;
    plot(vertex_closed_curve_min_only_reduced(:,2),vertex_closed_curve_min_only_reduced(:,1),'-go','MarkerFaceColor','g');
    plot(vertex_closed_curve_reduced(:,2),vertex_closed_curve_reduced(:,1),'-ro','MarkerFaceColor','r');
    axis equal;
end
title(['number of original vertices = ' num2str(size(vertex_closed_curve,1)) ' reduced vertices = ' num2str(size(vertex_closed_curve_reduced,1)) ])
hold off;

% Save reduced curves for later use
for idx_curve = 1:number_of_curves
    current_curve = all_closed_curves.(['Reduced_Curve_All_' num2str(idx_curve)]);
%     dlmwrite([OutPutPath 'Curve_' num2str(idx_curve) '_' OutputFileName '.txt'],current_curve);
    dlmwrite([OutPutPath OutPutName '_' num2str(idx_curve) '.txt'],current_curve);
end


all_closed_curves_split = struct;
% Split curves to reduce number of vertices per curve
number_of_curves_split = 0;
idx_curve_to_be_split = [];
number_of_new_curves = [];
for idx_curve = 1:number_of_curves
    size_of_current_curve = size(all_closed_curves.(['Reduced_Curve_All_' num2str(idx_curve)]),1);
    disp(['Size current curve: ' num2str(size_of_current_curve)]);
    if size_of_current_curve > max_number_of_vertices_per_curve
        number_of_curves_split = number_of_curves_split+1;
        idx_curve_to_be_split(number_of_curves_split) = idx_curve;
        number_of_new_curves(number_of_curves_split) = ceil(size_of_current_curve / max_number_of_vertices_per_curve);
    end
    
end

number_of_curves_split=0;
for idx_curve = idx_curve_to_be_split
    Current_curve_vertices = all_closed_curves.(['Reduced_Curve_All_' num2str(idx_curve)]);
    size_of_current_curve = size(all_closed_curves.(['Reduced_Curve_All_' num2str(idx_curve)]),1);
    number_of_curves_split=number_of_curves_split+1;
    for idx_new_curve = 1:number_of_new_curves(number_of_curves_split)
        idx_start = 1+(idx_new_curve-1)*max_number_of_vertices_per_curve;
        idx_stop = min([idx_new_curve*max_number_of_vertices_per_curve+1,size_of_current_curve]);
        if idx_new_curve ~= number_of_new_curves
            all_closed_curves_split.(['Split_Reduced_Curve_All_' num2str(idx_curve) '_new_idx_' num2str(idx_new_curve)]) = Current_curve_vertices(idx_start:idx_stop,:);
        else
            all_closed_curves_split.(['Split_Reduced_Curve_All_' num2str(idx_curve) '_new_idx_' num2str(idx_new_curve)]) = [Current_curve_vertices(idx_start:idx_stop,:);Current_curve_vertices(1,:)];
        end
    end
end

% Save split curves for later use
number_of_curves_split=0;
for idx_curve = idx_curve_to_be_split
    number_of_curves_split=number_of_curves_split+1;
    for idx_new_curve = 1:number_of_new_curves(number_of_curves_split)
        current_curve = all_closed_curves_split.(['Split_Reduced_Curve_All_' num2str(idx_curve) '_new_idx_' num2str(idx_new_curve)]);
        dlmwrite([OutPutPath OutPutName '_' num2str(idx_curve) '_Split_' num2str(idx_new_curve) '_' OutputFileName '.txt'],current_curve);
    end
end

















