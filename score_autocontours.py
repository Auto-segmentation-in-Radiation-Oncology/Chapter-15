import pydicom
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing
from shapely.ops import cascaded_union
from shapely.ops import split
import numpy as np
from scipy import stats as spstats
# import polygon_plot
import csv
import random
import json


def get_distance_measures(ref_poly, test_poly, stepsize=1.0, warningsize=1.0):
    # Hausdorff is trivial to compute with Shapely, but average distance requires stepping along each polygon.
    # This is the 'stepsize' in mm. At each point the minimum distance to the other contour is calculated to
    # create a list of distances. From this list the HD can be calculated from this, but it is inaccurate. Therefore,
    # we compare it to the Shapely one and report a problem if the error is greater that 'warningsize' in mm.

    reference_line = ref_poly.boundary
    test_line = test_poly.boundary

    distance_ref_to_test = []
    for distance_along_contour in np.arange(0, reference_line.length, stepsize):
        distance_to_other = reference_line.interpolate(distance_along_contour).distance(test_line)
        distance_ref_to_test.append(distance_to_other)

    distance_test_to_ref = []
    for distance_along_contour in np.arange(0, test_line.length, stepsize):
        distance_to_other = test_line.interpolate(distance_along_contour).distance(reference_line)
        distance_test_to_ref.append(distance_to_other)

    my_hd = np.max([np.max(distance_ref_to_test), np.max(distance_test_to_ref)])
    shapely_hd = test_poly.hausdorff_distance(ref_poly)

    if (my_hd + warningsize < shapely_hd) | (my_hd - warningsize > shapely_hd):
        print('There is a discrepancy between the Hausdorff distance and the list used to calculate the 95% HD')
        print('You may wish to consider a smaller stepsize')

    return distance_ref_to_test, distance_test_to_ref


def get_added_path_length(ref_poly, contracted_poly, expanded_poly, debug=False):

    added_path_length = 0

    reference_boundary = ref_poly.boundary
    if contracted_poly.area > 0:
        contracted_boundary = contracted_poly.boundary
    else:
        contracted_boundary = None
    expanded_boundary = expanded_poly.boundary

    # if debug:
    #     polygon_plot.plot_polygons_and_linestrings(reference_boundary, '#000000')
    #     if contracted_boundary is not None:
    #         polygon_plot.plot_polygons_and_linestrings(contracted_boundary, '#0000ff')
    #     polygon_plot.plot_polygons_and_linestrings(expanded_boundary, '#0000ff')

    if contracted_boundary is not None:
        split_success = False
        split_attempts = 0
        while (not split_success) & (split_attempts < 5):
            try:
                ref_split_inside = split(reference_boundary, contracted_boundary)
                split_success = True
            except ValueError:
                # Error can occur if sections parallel. Try a small jitter?
                contracted_poly_new = cascaded_union(contracted_poly.buffer(random.random() * 0.0001, 32, 1, 1))
                contracted_boundary = contracted_poly_new.boundary
                split_attempts = split_attempts + 1

        if split_success:
            for line_segment in ref_split_inside:
                # check it the centre of the line is within the contracted polygon
                mid_point = line_segment.interpolate(0.5, True)
                if contracted_poly.contains(mid_point):
                    added_path_length = added_path_length + line_segment.length
                #     if debug:
                #         polygon_plot.plot_polygons_and_linestrings(line_segment, '#00ff00')
                # else:
                #     if debug:
                #         polygon_plot.plot_polygons_and_linestrings(line_segment, '#ff0000')
        else:
            if debug:
                print('Failed to correctly calculate Added Path Length for a slice of an organ', flush=True)
                # would be nice if we had the information to return here!

    split_success = False
    split_attempts = 0
    while (not split_success) & (split_attempts < 5):
        try:
            ref_split_outside = split(reference_boundary, expanded_boundary)
            split_success = True
        except ValueError:
            # Error can occur if sections parallel. Try a tiny random jitter
            expanded_poly_new = cascaded_union(expanded_poly.buffer(random.random() * 0.0001, 32, 1, 1))
            expanded_boundary = expanded_poly_new.boundary
            split_attempts = split_attempts + 1

    if split_success:
        for line_segment in ref_split_outside:
            # check it the centre of the line is outside the expanded polygon
            mid_point = line_segment.interpolate(0.5, True)
            if not expanded_poly.contains(mid_point):
                added_path_length = added_path_length + line_segment.length
            #     if debug:
            #         polygon_plot.plot_polygons_and_linestrings(line_segment, '#00ff00')
            # else:
            #     if debug:
            #         polygon_plot.plot_polygons_and_linestrings(line_segment, '#ff0000')
    else:
        if debug:
            print('Failed to correctly calculate Added Path Length for a slice of an organ', flush=True)
            # would be nice if we had the information to return here!

    return added_path_length


def find_and_score_slice_matches(ref_rtss, test_rtss, slice_thickness, contour_matches, tolerance=1.0, debug=False):
    result_list = []

    # for each structure if match list
    for idx, match_ids in enumerate(contour_matches):

        ref_id, test_id = match_ids
        total_added_path_length = 0
        test_contour_length = 0
        ref_contour_length = 0
        total_true_positive_area = 0
        total_false_positive_area = 0
        total_false_negative_area = 0
        total_test_area = 0
        total_ref_area = 0
        distance_ref_to_test = []
        distance_test_to_ref = []
        ref_weighted_centroid_sum = np.array([0, 0, 0])
        test_weighted_centroid_sum = np.array([0, 0, 0])
        structure_name = ''

        # get the structure name just for fun
        for ref_set in ref_rtss.StructureSetROISequence:
            if ref_set.ROINumber == ref_id:
                structure_name = ref_set.ROIName
                break

        print('Computing scores for: ', structure_name)

        ref_contour_set = None
        test_contour_set = None

        # Find contour set for reference and test
        for contour_set in ref_rtss.ROIContourSequence:
            if contour_set.ReferencedROINumber == ref_id:
                ref_contour_set = contour_set
                break
        for contour_set in test_rtss.ROIContourSequence:
            if contour_set.ReferencedROINumber == test_id:
                test_contour_set = contour_set
                break

        ref_polygon_dictionary, ref_z_slices = build_polygon_dictionary(ref_contour_set)
        test_polygon_dictionary, test_z_slices = build_polygon_dictionary(test_contour_set)

        # for each slice in ref find corresponding slice in test
        for z_value, refpolygon in ref_polygon_dictionary.items():
            if z_value in test_z_slices:
                testpolygon = test_polygon_dictionary[z_value]

                # if (structure_name == 'Lung_L') & (ref_contour[2] == -328):
                #    print(ref_contour[2])
                #    debug = True

                # if debug:
                #     polygon_plot.plot_polygons_and_linestrings(refpolygon, '#00ff00')
                #     polygon_plot.plot_polygons_and_linestrings(testpolygon, '#0000ff')

                # go get some distance measures
                # these get added to a big list so that we can calculate the 95% HD
                [ref_to_test, test_to_ref] = get_distance_measures(refpolygon, testpolygon, 0.05)
                distance_ref_to_test.extend(ref_to_test)
                distance_test_to_ref.extend(test_to_ref)

                # apply tolerance ring margin to test with added path length
                expanded_poly = cascaded_union(testpolygon.buffer(tolerance, 32, 1, 1))
                contracted_poly = cascaded_union(testpolygon.buffer(-tolerance, 32, 1, 1))

                # if debug:
                #     polygon_plot.plot_polygons_and_linestrings(expanded_poly, '#ff0000')
                #     polygon_plot.plot_polygons_and_linestrings(contracted_poly, '#ff0000')

                # add intersection of contours
                contour_intersection = refpolygon.intersection(testpolygon)
                total_true_positive_area = total_true_positive_area + contour_intersection.area
                total_false_negative_area = total_false_negative_area + \
                                            (refpolygon.difference(contour_intersection)).area
                total_false_positive_area = total_false_positive_area + \
                                            (testpolygon.difference(contour_intersection)).area
                total_test_area = total_test_area + testpolygon.area
                total_ref_area = total_ref_area + refpolygon.area
                centroid_point = refpolygon.centroid
                centroid_point_np = np.array([centroid_point.x, centroid_point.y, z_value])
                ref_weighted_centroid_sum = ref_weighted_centroid_sum + (refpolygon.area * centroid_point_np)
                centroid_point = testpolygon.centroid
                centroid_point_np = np.array([centroid_point.x, centroid_point.y, z_value])
                test_weighted_centroid_sum = test_weighted_centroid_sum + (testpolygon.area * centroid_point_np)

                # if debug == True:
                #    x, y = Polygon(LinearRing(ref_contour_2_d)).boundary.xy
                #    plt.plot(x, y, color='#ff0000')
                #    x, y = expanded_poly.boundary.xy
                #    plt.plot(x, y, color='#00ff00')
                #    x, y = contracted_poly.boundary.xy
                #    plt.plot(x, y, color='#0000ff')

                # add length of remain contours
                added_path = get_added_path_length(refpolygon, contracted_poly, expanded_poly, debug=debug)
                total_added_path_length = total_added_path_length + added_path
                test_contour_length = test_contour_length + testpolygon.length
                ref_contour_length = ref_contour_length + refpolygon.length

            else:
                # if no corresponding slice, then add the whole ref length
                # print('Adding path for whole contour')
                path_length = refpolygon.length
                total_added_path_length = total_added_path_length + path_length
                ref_contour_length = ref_contour_length + refpolygon.length
                # also the whole slice is false negative
                total_false_negative_area = total_false_negative_area + refpolygon.area
                total_ref_area = total_ref_area + refpolygon.area
                centroid_point = refpolygon.centroid
                centroid_point_np = np.array([centroid_point.x, centroid_point.y, z_value])
                ref_weighted_centroid_sum = ref_weighted_centroid_sum + (refpolygon.area * centroid_point_np)

        # we also need to consider the slices for which there is a test contour but no reference
        for z_value, testpolygon in test_polygon_dictionary.items():
            if z_value not in ref_z_slices:
                # add path length doesn't get updated
                test_contour_length = test_contour_length + testpolygon.length
                # but the whole slice is false positive
                total_false_positive_area = total_false_positive_area + testpolygon.area
                total_test_area = total_test_area + testpolygon.area
                centroid_point = testpolygon.centroid
                centroid_point_np = np.array([centroid_point.x, centroid_point.y, z_value])
                test_weighted_centroid_sum = test_weighted_centroid_sum + (testpolygon.area * centroid_point_np)

        # now we need to deal with the distance lists to work out the various distance measures
        # NOTE: these are different calculations to those used in plastimatch. The book chapter will explain all..

        # Added the test to avoid division bt zeros for empty structures.  We should have avoid it from happening before
        if total_ref_area > 0:
            ref_centroid = ref_weighted_centroid_sum / total_ref_area
        else:
            ref_centroid = np.array([0, 0, 0])
        if total_test_area > 0:
            test_centroid = test_weighted_centroid_sum / total_test_area
        else:
            test_centroid = np.array([0, 0, 0])

        if distance_ref_to_test == [] and distance_test_to_ref == []:
            if debug:
                print('Empty contours or are not on the same slices!')
            hd = float('nan')
            ninety_five_hd = float('nan')
            ave_dist = float('nan')
            median_dist = float('nan')
        else:
            hd = np.max([np.max(distance_ref_to_test), np.max(distance_test_to_ref)])
            ninety_five_hd = np.max([np.percentile(distance_ref_to_test, 95), np.percentile(distance_test_to_ref, 95)])
            ave_dist = (np.mean(distance_ref_to_test) + np.mean(distance_test_to_ref)) / 2
            median_dist = (np.median(distance_ref_to_test) + np.median(distance_test_to_ref)) / 2

        # print('added path length = ', total_added_path_length)
        result_list.append((structure_name,
                            [total_added_path_length, ref_contour_length, test_contour_length,
                             total_true_positive_area * slice_thickness,
                             total_false_negative_area * slice_thickness,
                             total_false_positive_area * slice_thickness, total_ref_area * slice_thickness,
                             total_test_area * slice_thickness,
                             hd, ninety_five_hd, ave_dist, median_dist, ref_centroid, test_centroid]))

    return result_list


def build_polygon_dictionary(contour_set):
    # this function extracts the polygon data from the pydicom structure and
    # converts it to a dictionary of Shapely polygons by the z slice
    # TODO: Ideally this function could be improved for off-axis data.

    polygon_dictionary = {}
    z_slices = []
    if contour_set is not None:
        # get the list of z-values for the reference set
        for contour_slice in contour_set.ContourSequence:
            number_pts = int(contour_slice.NumberOfContourPoints)
            if number_pts >= 3:  # We check for zero volume/level at the slice level
                contour_points = contour_slice.ContourData
                z_slices.append(contour_points[2])
        # round to 1 decimal place (0.1mm) to make finding a match more robust
        z_slices = np.round(z_slices, 1)
        z_slices = np.unique(z_slices)

        # now build the multi-polygon for each z-slice
        for z_value in z_slices:
            polygon_data = None
            for contour_slice in contour_set.ContourSequence:
                number_pts = int(contour_slice.NumberOfContourPoints)
                if number_pts >= 3:
                    contour_points = contour_slice.ContourData
                    if np.round(contour_points[2], 1) == z_value:
                        # make 2D contours
                        contour_points_2d = np.zeros((number_pts, 2))
                        for i in range(0, number_pts):
                            contour_points_2d[i][0] = float(contour_points[i * 3])
                            contour_points_2d[i][1] = float(contour_points[i * 3 + 1])
                        if polygon_data is None:
                            # Make points into Polygon
                            polygon_data = Polygon(LinearRing(contour_points_2d))
                        else:
                            # Turn next set of points into a Polygon
                            current_polygon = Polygon(LinearRing(contour_points_2d))
                            # Attempt to fix any self-intersections in the resulting polygon
                            if not current_polygon.is_valid:
                                current_polygon = current_polygon.buffer(0)
                            if polygon_data.contains(current_polygon):
                                # if the new polygon is inside the old one, chop it out
                                polygon_data = polygon_data.difference(current_polygon)
                            elif polygon_data.within(current_polygon):
                                # if the new and vice versa
                                polygon_data = current_polygon.difference(polygon_data)
                            else:
                                # otherwise it is a floating blob to add
                                polygon_data = polygon_data.union(current_polygon)
                        # Attempt to fix any self-intersections in the resulting polygon
                        if polygon_data is not None:
                            if not polygon_data.is_valid:
                                polygon_data = polygon_data.buffer(0)
            # check this slice has a tangible size polygon.
            if (polygon_data.length > 0) & (polygon_data.area > 0):
                polygon_dictionary[z_value] = polygon_data
    return polygon_dictionary, z_slices


def estimate_slice_thickness(contour_data_set, debug=False):
    # this is a crude attempt to estimate the slice thickness without loading the image
    # we assume that the slices are equally spaced, and if we collect unique slice positions
    # for enough slices with contours then the modal difference will represent the slice thickness

    z_list = []
    z_diff_list = []

    for contour_set in contour_data_set.ROIContourSequence:
        if hasattr(contour_set, 'ContourSequence'):
            for contour_slice in contour_set.ContourSequence:
                contour_points = contour_slice.ContourData
                z_list.append(contour_points[2])

    z_list = np.unique(z_list)
    z_list = np.sort(z_list)

    old_z_val = z_list[0]
    for z_val in z_list:
        z_diff = z_val - old_z_val
        old_z_val = z_val
        z_diff_list.append(z_diff)

    slice_thickness = spstats.mode(z_diff_list)[0][0]
    if debug:
        print('slice thickness: ', slice_thickness)

    return slice_thickness


def score_case(reference_rtss_filename, test_rtss_filename, slice_thickness=0, output_filename='', tolerance=1.0):
    # load the DICOM files
    ground_truth_data = pydicom.read_file(reference_rtss_filename, False)
    test_data = pydicom.read_file(test_rtss_filename, False)

    if slice_thickness == 0:
        slice_thickness = estimate_slice_thickness(ground_truth_data)

    # find the matching structure names
    contour_matches = []
    for ref_roi in ground_truth_data.StructureSetROISequence:

        ref_name = ref_roi.ROIName
        ref_id = ref_roi.ROINumber
        ref_contour_set = None

        # Find contour set for reference
        for contour_set in ground_truth_data.ROIContourSequence:
            if contour_set.ReferencedROINumber == ref_id:
                ref_contour_set = contour_set
                break

        # Don't bother checking for a match if the reference is empty
        # TODO could also check for 0 length or 0 volume, but that is more effort
        if ref_contour_set is None:
            print(' Reference contour is empty for structure', ref_name)
            continue

        if hasattr(ref_contour_set, 'ContourSequence'):
            number_of_matches = 0
            print('Checking structure: {:s}'.format(ref_name))

            last_match_id = -1
            for test_roi in test_data.StructureSetROISequence:
                test_name = test_roi.ROIName

                if test_name.lower().strip() == ref_name.lower().strip():
                    number_of_matches = number_of_matches + 1
                    last_match_id = test_roi.ROINumber

            if number_of_matches == 1:
                contour_matches.append((ref_id, last_match_id))
            elif number_of_matches == 0:
                print('\tNo match for structure: {:s}\n\tSkipping structure'.format(ref_name))
            elif number_of_matches > 1:
                # TODO compare to each and report for all?
                print('\tMultiple matches for structure: {:s}\n\tSkipping structure'.format(ref_name))

    resultlist = find_and_score_slice_matches(ground_truth_data, test_data, slice_thickness, contour_matches,
                                              tolerance=tolerance)

    auto_contour_measures = format_result_list(resultlist)

    if output_filename != '':
        print('Writing results to: ', output_filename)
        with open(output_filename, mode='w', newline='\n', encoding='utf-8') as out_file:
            result_writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            result_writer.writerow(
                ['Organ', 'APL', 'TP volume', 'FN volume', 'FP volume', 'SEN', '%FP', '3D DSC', '2D HD',
                 '95% 2D HD', 'Ave 2D Dist', 'Median 2D Dist', 'Reference Centroid', 'Test Centroid'])
            for results_structure in auto_contour_measures:
                result_writer.writerow([results_structure['Organ'],
                                        results_structure['APL'],
                                        results_structure['TPVol'],
                                        results_structure['FNVol'],
                                        results_structure['FPVol'],
                                        results_structure['SEN'],
                                        results_structure['SFP'],
                                        results_structure['three_D_DSC'],
                                        results_structure['HD'],
                                        results_structure['ninety_five_HD'],
                                        results_structure['AD'],
                                        results_structure['MD'],
                                        results_structure['ref_cent'],
                                        results_structure['test_cent']])
    else:
        # TODO function could take list of parameters of measures we want to return.
        return auto_contour_measures


def format_result_list(result_list):
    formatted_results = []
    for result in result_list:
        organ_name, scores = result
        # scores[0] APL
        # scores[1] Ref PL
        # scores[2] Test PL
        # scores[3] TP volume
        # scores[4] FN volume
        # scores[5] FP volume
        # scores[6] Ref volume
        # scores[7] Test volume
        # scores[8] Hausdorff
        # scores[9] 95% Hausdorff
        # scores[10] Average Distance
        # scores[11] Median Distance
        # scores[12] Reference Centroid
        # scores[13] Test Centroid
        results_structure = {'Organ': organ_name, 'APL': scores[0], 'ref_length': scores[1], 'test_length': scores[2],
                             'TPVol': scores[3], 'FNVol': scores[4],
                             'FPVol': scores[5], 'ref_vol': scores[6], 'test_vol': scores[7],
                             'SEN': scores[3] / scores[6], 'FPFrac': scores[5] / scores[6],
                             'Inclusiveness': scores[3] / scores[7], 'PPV': scores[3]/(scores[3]+scores[5]),
                             'three_D_DSC': 2 * scores[3] / (scores[6] + scores[7]), 'HD': scores[8],
                             'ninety_five_HD': scores[9], 'AD': scores[10], 'MD': scores[11], 'ref_cent': scores[12],
                             'test_cent': scores[13], 'cent_dist': np.linalg.norm(scores[12]-scores[13])}
        formatted_results.append(results_structure)
    return formatted_results


