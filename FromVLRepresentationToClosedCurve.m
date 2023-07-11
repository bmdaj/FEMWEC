%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Converts the representation format of the closed curve from vertex-line
% to array of vertices.
%
% Written by: REC@MEK.DTU.DK in year 2018
%
% Disclaimer:                                                              
% The author reserves all rights but does not guaranty that the code is   
% free from errors. Furthermore, we shall not be liable in any event     
% caused by the use of the program.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ all_closed_curves ] = FromVLRepresentationToClosedCurve( Vertices, Lines, curve_index_list )

all_closed_curves = struct;
number_of_curves = max(curve_index_list);
for idx_curve = 1:number_of_curves
    Lines_on_curve =  Lines(find(curve_index_list==idx_curve),:);
    % Sort vertices to single connected curve. (assuming one feature)
    vertex_1 = Lines_on_curve(1,1);
    vertex_2 = Lines_on_curve(1,2);
    number_of_lines = size(Lines_on_curve,1);
    vertex_closed_curve = zeros(number_of_lines+1,2);
    for idx_lines = 1:number_of_lines
        % Store first vertex on line in vertex_closed_curve
        vertex_closed_curve(idx_lines,:) = Vertices(vertex_1,:);
        
        % Find second vertex on next line segment on curve
        vertex_1 = vertex_2;
        vertex_2 = Lines_on_curve(find(Lines_on_curve(:,1)==vertex_2),2);
        
    end
    vertex_closed_curve(end,:) = vertex_closed_curve(1,:);
    
    all_closed_curves.(['Curve_' num2str(idx_curve)]) = vertex_closed_curve;
end


end

