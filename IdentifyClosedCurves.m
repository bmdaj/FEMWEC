function [ curve_index_list, number_of_curves ] = IdentifyClosedCurves( Lines )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Identifies and creates index lists for closed curves contained in 
% the Nx2 array of indices for N line segments contained in Lines.
%
% Input: 
% Lines [Nx2 integers] : Line segments 
% 
% Written by: REC@MEK.DTU.DK 2018
%
% Disclaimer:                                                              
% The author reserves all rights but does not guaranty that the code is   
% free from errors. Furthermore, we shall not be liable in any event     
% caused by the use of the program.
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Identify the number of curves and associated lines and vertices
Lines_reduced = Lines;
number_of_lines = size(Lines,1);
curve_index_list = zeros(number_of_lines,1);
vertex_1 = Lines_reduced(1,1);
vertex_2 = Lines_reduced(1,2);
idx_curve = 1; 
while (sum(curve_index_list==0)>0)
    % Store first vertex on line in vertex_closed_curve
    curve_index_list(find(Lines(:,1)==vertex_1)) = idx_curve;
    
    % Find second vertex on next line segment on curve
    vertex_1 = vertex_2;
    if isempty(find(Lines(:,1)==vertex_2))==1
        error('Open curve identified. Terminating!');
    end
    vertex_2 = Lines(find(Lines(:,1)==vertex_2),2);
    
    % Check if a closed curve has been traversed. If so jump to a new curve
    % contained in Lines.
    if (curve_index_list(find(Lines(:,1)==vertex_1)) == idx_curve)
        Lines_reduced = Lines(find(curve_index_list==0),:);
        if isempty(Lines_reduced)~=1
        idx_curve=idx_curve+1;
        vertex_1 = Lines_reduced(1,1);
        vertex_2 = Lines_reduced(1,2);
        end
    end
end
number_of_curves = idx_curve;


end

