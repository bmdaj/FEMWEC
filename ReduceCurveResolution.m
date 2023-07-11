%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Reduces the number of points constituting a closed curve according to
% user specified input parameters.
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
function [ all_closed_curves ] = ReduceCurveResolution( number_of_curves, all_closed_curves, ModelReduction )

critical_angle = ModelReduction.critical_angle;
max_vertex_distance = ModelReduction.max_vertex_distance;
min_vertex_distance = ModelReduction.min_vertex_distance;
number_of_vertices_jumped_in_check = ModelReduction.number_of_vertices_jumped_in_check;


% Reducing the vertices on the curve considered in the following based on
% their distance to neighbors (along the curve)
for idx_curve = 1:number_of_curves
    number_of_lines =  size(all_closed_curves.(['Curve_' num2str(idx_curve)]),1)-1;
    vertex_remove_list = zeros(number_of_lines+1,1);
    number_of_vertices_removed = 1;
    vertex_closed_curve_reduced = all_closed_curves.(['Curve_' num2str(idx_curve)]);
    while number_of_vertices_removed>0
        vertex_distance=0;
        idx_lines=2;
        idx_lines_tmp=1;
        number_of_vertices_removed=0;
        while (idx_lines <= number_of_lines-1)
            vector_current_to_next_vertex = diff(vertex_closed_curve_reduced(idx_lines_tmp:idx_lines_tmp+1,:));
            vertex_distance = vertex_distance+norm(vector_current_to_next_vertex,2);
            if  (vertex_distance<min_vertex_distance)
                vertex_remove_list(idx_lines_tmp+1) = 1;
                idx_lines_tmp=idx_lines_tmp+1;
                idx_lines=idx_lines_tmp+1;
                number_of_vertices_removed=number_of_vertices_removed+1;
            else
                idx_lines=idx_lines+1;
                idx_lines_tmp=idx_lines_tmp+1;
                vertex_distance=0;
            end
        
        end
        % Checking if final vertex should be removed
        vector_current_to_next_vertex = diff([vertex_closed_curve_reduced(end-1,:);vertex_closed_curve_reduced(end,:)]);
        vertex_distance = vertex_distance+norm(vector_current_to_next_vertex,2);
        if  (vertex_distance<min_vertex_distance)
            vertex_remove_list(idx_lines_tmp+1) = 1;
            idx_lines_tmp=idx_lines_tmp+1;
            idx_lines=idx_lines_tmp+1;
            number_of_vertices_removed=number_of_vertices_removed+1;
        end
        
        % Storing data
        vertex_closed_curve_reduced = vertex_closed_curve_reduced(vertex_remove_list==0,:);
        number_of_lines=size(vertex_closed_curve_reduced,1);
        vertex_remove_list = zeros(size(vertex_closed_curve_reduced,1),1);
        number_of_vertices_removed=0;
    end
    all_closed_curves.(['Reduced_Curve_Min_Only_' num2str(idx_curve)]) = vertex_closed_curve_reduced;
end

% Removing vertices from the reduced curves based on angular and maximum
% distance constraints
for idx_curve = 1:number_of_curves
    number_of_lines =  size(all_closed_curves.(['Reduced_Curve_Min_Only_' num2str(idx_curve)]),1)-1;
    vertex_remove_list = zeros(number_of_lines+1,1);
    number_of_vertices_removed = 1;
    vertex_closed_curve_reduced = all_closed_curves.(['Reduced_Curve_Min_Only_' num2str(idx_curve)]);
    while number_of_vertices_removed>0
        idx_lines=1;
        idx_lines_tmp=1;
        idx_counter = 0;
        number_of_vertices_removed=0;
        while (idx_lines <= number_of_lines-(number_of_vertices_jumped_in_check+idx_counter))
            vector_1 = diff(vertex_closed_curve_reduced(idx_lines_tmp:idx_lines+1,:));
            vector_2 = diff(vertex_closed_curve_reduced(idx_lines_tmp:(number_of_vertices_jumped_in_check+idx_counter):idx_lines+(number_of_vertices_jumped_in_check+idx_counter),:));
            
            angle_un_mod = real(acos((vector_1(2)*vector_2(2)+vector_1(1)*vector_2(1))/(norm(vector_1,2)*norm(vector_2,2))));
            angle_between_vectors = (mod(angle_un_mod,2*pi));
%             if (vertex_closed_curve_reduced(idx_lines_tmp,1)<2)
%                 disp(['y: ' num2str(vertex_closed_curve_reduced(idx_lines_tmp,1)) ' x: ' num2str(vertex_closed_curve_reduced(idx_lines_tmp,2))]);
%                 disp(['ang: ' num2str(angle_between_vectors)]);
%                 rekeyboard;
%             end
            vertex_distance = norm(vector_2,2);
            
            if (angle_between_vectors<critical_angle) && (vertex_distance<max_vertex_distance)
                vertex_remove_list(idx_lines+1) = 1;
                idx_lines=idx_lines+1;
                number_of_vertices_removed=number_of_vertices_removed+1;
                idx_counter=0;
            else
                idx_lines=idx_lines+1;
                idx_lines_tmp = idx_lines;
                idx_counter=idx_counter+1;
                idx_counter=0;
            end
            
        end
        if number_of_vertices_removed>0
            vertex_closed_curve_reduced = vertex_closed_curve_reduced(vertex_remove_list==0,:);
            number_of_lines=size(vertex_closed_curve_reduced,1);
            vertex_remove_list = zeros(size(vertex_closed_curve_reduced,1),1);
        end
    end
    all_closed_curves.(['Reduced_Curve_All_' num2str(idx_curve)]) = vertex_closed_curve_reduced;
end

end

