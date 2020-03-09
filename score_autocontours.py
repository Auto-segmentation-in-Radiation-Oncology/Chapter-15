import pydicom
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing
from shapely.ops import cascaded_union
from shapely.ops import split
import numpy as np
from scipy import stats as spstats
import polygon_plot
import csv


# https://stackoverflow.com/questions/56965268/how-do-i-convert-a-3d-point-cloud-ply-into-a-mesh-with-faces-and-vertices

############################################################################################


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

    total_path_length = 0

    reference_boundary = ref_poly.boundary
    if contracted_poly.area > 0:
        contracted_boundary = contracted_poly.boundary
    else:
        contracted_boundary = None
    expanded_boundary = expanded_poly.boundary

    if debug:
        polygon_plot.plot_polygons_and_linestrings(reference_boundary, '#000000')
        if contracted_boundary is not None:
            polygon_plot.plot_polygons_and_linestrings(contracted_boundary, '#0000ff')
        polygon_plot.plot_polygons_and_linestrings(expanded_boundary, '#0000ff')

    if contracted_boundary is not None:
        ref_split_inside = split(reference_boundary, contracted_boundary)
        for line_segment in ref_split_inside:
            # check it the centre of the line is within the contracted polygon
            mid_point = line_segment.interpolate(0.5, True)
            if contracted_poly.contains(mid_point):
                total_path_length = total_path_length + line_segment.length
                if debug:
                    polygon_plot.plot_polygons_and_linestrings(line_segment, '#00ff00')
            else:
                if debug:
                    polygon_plot.plot_polygons_and_linestrings(line_segment, '#ff0000')

    ref_split_outside = split(reference_boundary, expanded_boundary)
    for line_segment in ref_split_outside:
        # check it the centre of the line is outside the expanded polygon
        mid_point = line_segment.interpolate(0.5, True)
        if not expanded_poly.contains(mid_point):
            total_path_length = total_path_length + line_segment.length
            if debug:
                polygon_plot.plot_polygons_and_linestrings(line_segment, '#00ff00')
        else:
            if debug:
                polygon_plot.plot_polygons_and_linestrings(line_segment, '#ff0000')

    return total_path_length


def find_and_score_slice_matches(ref_rtss, test_rtss, slice_thickness, contour_matches, tolerance=1):
    result_list = []

    # for each structure if match list
    for idx, match_ids in enumerate(contour_matches):

        ref_id, test_id = match_ids
        total_added_path_length = 0
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

        ref_polygon_dictionary = {}
        test_polygon_dictionary = {}
        ref_z_slices = []
        test_z_slices = []

        if ref_contour_set is not None:
            # get the list of z-values for the reference set
            for ref_contour_slice in ref_contour_set.ContourSequence:
                n_ref_pts = int(ref_contour_slice.NumberOfContourPoints)
                if n_ref_pts >= 4:
                    ref_contour = ref_contour_slice.ContourData
                    ref_z_slices.append(ref_contour[2])
            # round to 1 decimal place (0.1mm) to make finding a match more robust
            ref_z_slices = np.round(ref_z_slices, 1)
            ref_z_slices = np.unique(ref_z_slices)

            # now build the multi-polygon for each z-slice
            for z_value in ref_z_slices:
                ref_polygon = None
                for ref_contour_slice in ref_contour_set.ContourSequence:
                    n_ref_pts = int(ref_contour_slice.NumberOfContourPoints)
                    if n_ref_pts >= 4:
                        ref_contour = ref_contour_slice.ContourData
                        if np.round(ref_contour[2], 1) == z_value:
                            # make 2D contours
                            ref_contour_2_d = np.zeros((n_ref_pts, 2))
                            for i in range(0, n_ref_pts):
                                ref_contour_2_d[i][0] = float(ref_contour[i * 3])
                                ref_contour_2_d[i][1] = float(ref_contour[i * 3 + 1])
                            if ref_polygon is None:
                                # Make points into Polygon
                                ref_polygon = Polygon(LinearRing(ref_contour_2_d))
                            else:
                                # Turn next set of points into a Polygon
                                this_ref_polygon = Polygon(LinearRing(ref_contour_2_d))
                                # Attempt to fix any self-intersections in the resulting polygon
                                if not this_ref_polygon.is_valid:
                                    this_ref_polygon = this_ref_polygon.buffer(0)
                                if ref_polygon.contains(this_ref_polygon):
                                    # if the new polygon is inside the old one, chop it out
                                    ref_polygon = ref_polygon.difference(this_ref_polygon)
                                elif ref_polygon.within(this_ref_polygon):
                                    # if the new and vice versa
                                    ref_polygon = this_ref_polygon.difference(ref_polygon)
                                else:
                                    # otherwise it is a floating blob to add
                                    ref_polygon = ref_polygon.union(this_ref_polygon)
                            # Attempt to fix any self-intersections in the resulting polygon
                            if ref_polygon is not None:
                                if not ref_polygon.is_valid:
                                    ref_polygon = ref_polygon.buffer(0)
                ref_polygon_dictionary[z_value] = ref_polygon

        # get the list of z-values for the reference set
        for test_contour_slice in test_contour_set.ContourSequence:
            n_test_pts = int(test_contour_slice.NumberOfContourPoints)
            if n_test_pts >= 4:
                test_contour = test_contour_slice.ContourData
                test_z_slices.append(test_contour[2])
        test_z_slices = np.round(test_z_slices, 1)
        test_z_slices = np.unique(test_z_slices)

        if test_contour_set is not None:
            # now build the multi-polygon for each z-slice
            for z_value in test_z_slices:
                test_polygon = None
                for test_contour_slice in test_contour_set.ContourSequence:
                    n_test_pts = int(test_contour_slice.NumberOfContourPoints)
                    if n_test_pts >= 4:
                        test_contour = test_contour_slice.ContourData
                        if np.round(test_contour[2], 1) == z_value:
                            # make 2D contours
                            test_contour_2_d = np.zeros((n_test_pts, 2))
                            for i in range(0, n_test_pts):
                                test_contour_2_d[i][0] = float(test_contour[i * 3])
                                test_contour_2_d[i][1] = float(test_contour[i * 3 + 1])

                            if test_polygon is None:
                                # Make points into Polygon
                                test_polygon = Polygon(LinearRing(test_contour_2_d))
                            else:
                                # Turn next set of points into a Polygon
                                this_test_polygon = Polygon(LinearRing(test_contour_2_d))
                                # Attempt to fix any self-intersections
                                if not this_test_polygon.is_valid:
                                    this_test_polygon = this_test_polygon.buffer(0)
                                if test_polygon.contains(this_test_polygon):
                                    # if the new polygon is inside the old one, chop it out
                                    test_polygon = test_polygon.difference(this_test_polygon)
                                elif test_polygon.within(this_test_polygon):
                                    # if the new and vice versa
                                    test_polygon = this_test_polygon.difference(test_polygon)
                                else:
                                    # otherwise it is a floating blob to add
                                    test_polygon = test_polygon.union(this_test_polygon)

                            # Attempt to fix any self-intersections in the resulting polygon
                            if not test_polygon.is_valid:
                                test_polygon = test_polygon.buffer(0)
                test_polygon_dictionary[z_value] = test_polygon

        # for each slice in ref find corresponding slice in test
        for z_value, refpolygon in ref_polygon_dictionary.items():
            if z_value in test_z_slices:
                testpolygon = test_polygon_dictionary[z_value]

                debug = False

                # if (structure_name == 'Lung_L') & (ref_contour[2] == -328):
                #    print(ref_contour[2])
                #    debug = True

                if debug:
                    polygon_plot.plot_polygons_and_linestrings(refpolygon, '#00ff00')
                    polygon_plot.plot_polygons_and_linestrings(testpolygon, '#0000ff')

                # go get some distance measures
                # these get added to a big list so that we can calculate the 95% HD
                [ref_to_test, test_to_ref] = get_distance_measures(refpolygon, testpolygon, 0.05)
                distance_ref_to_test.extend(ref_to_test)
                distance_test_to_ref.extend(test_to_ref)

                # apply tolerance ring margin to test with added path length
                expanded_poly = cascaded_union(testpolygon.buffer(tolerance, 32, 1, 1))
                contracted_poly = cascaded_union(testpolygon.buffer(-tolerance, 32, 1, 1))

                if debug:
                    polygon_plot.plot_polygons_and_linestrings(expanded_poly, '#ff0000')
                    polygon_plot.plot_polygons_and_linestrings(contracted_poly, '#ff0000')

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

                added_path = get_added_path_length(refpolygon, contracted_poly, expanded_poly)
                total_added_path_length = total_added_path_length + added_path

            else:
                # if no corresponding slice, then add the whole ref length
                # print('Adding path for whole contour')
                path_length = refpolygon.length
                total_added_path_length = total_added_path_length + path_length
                # also the whole slice is false negative
                total_false_negative_area = total_false_negative_area + refpolygon.area
                total_ref_area = total_ref_area + refpolygon.area
                centroid_point = refpolygon.centroid
                centroid_point_np = np.array([centroid_point.x, centroid_point.y, z_value])
                ref_weighted_centroid_sum = ref_weighted_centroid_sum + (refpolygon.area * centroid_point_np)

        # we also need to consider the slices for which there is a test contour but no reference
        for z_value, testpolygon in test_polygon_dictionary.items():
            if z_value not in ref_z_slices:
                # path length doesn't get updated
                # but the whole slice is false positive
                total_false_positive_area = total_false_positive_area + testpolygon.area
                total_test_area = total_test_area + testpolygon.area
                centroid_point = testpolygon.centroid
                centroid_point_np = np.array([centroid_point.x, centroid_point.y, z_value])
                test_weighted_centroid_sum = test_weighted_centroid_sum + (testpolygon.area * centroid_point_np)

        # now we need to deal with the distance lists to work out the various distance measures
        # NOTE: these are different calculations to those used in plastimatch. The book chapter will explain all...

        ref_centroid = ref_weighted_centroid_sum / total_ref_area
        test_centroid = test_weighted_centroid_sum / total_test_area

        hd = np.max([np.max(distance_ref_to_test), np.max(distance_test_to_ref)])
        ninety_five_hd = np.max([np.percentile(distance_ref_to_test, 95), np.percentile(distance_test_to_ref, 95)])
        ave_dist = (np.mean(distance_ref_to_test) + np.mean(distance_test_to_ref)) / 2
        median_dist = (np.median(distance_ref_to_test) + np.median(distance_test_to_ref)) / 2

        # print('added path length = ', total_added_path_length)
        result_list.append((structure_name,
                            [total_added_path_length, total_true_positive_area * slice_thickness,
                             total_false_negative_area * slice_thickness,
                             total_false_positive_area * slice_thickness, total_ref_area * slice_thickness,
                             total_test_area * slice_thickness,
                             hd, ninety_five_hd, ave_dist, median_dist, ref_centroid, test_centroid]))

    return result_list


def estimate_slice_thickness(contour_data_set):
    # this is a crude attempt to estimate the slice thickness without loading the image
    # we assume that the slices are equally spaced, and if we collect unique slice positions
    # for enough slices with contours then the modal difference will represent the slice thickness

    z_list = []
    z_diff_list = []

    for contour_set in contour_data_set.ROIContourSequence:
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

    print('slice thickness: ', slice_thickness)

    return slice_thickness


def score_case(reference_rtss_filename, test_rtss_filename, slice_thickness=0, output_filename=''):
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
        if ref_contour_set is not None:
            if hasattr(ref_contour_set, 'ContourSequence'):
                number_of_matches = 0
                print('Checking structure: {:s}'.format(ref_name))

                last_match_id = -1
                for test_roi in test_data.StructureSetROISequence:
                    test_name = test_roi.ROIName

                    if test_name == ref_name:
                        number_of_matches = number_of_matches + 1
                        last_match_id = test_roi.ROINumber

                if number_of_matches == 1:
                    contour_matches.append((ref_id, last_match_id))
                elif number_of_matches == 0:
                    print('\tNo match for structure: {:s}\n\tSkipping structure'.format(ref_name))
                elif number_of_matches > 1:
                    # TODO compare to each and report for all?
                    print('\tMultiple matches for structure: {:s}\n\tSkipping structure'.format(ref_name))

    resultlist = find_and_score_slice_matches(ground_truth_data, test_data, slice_thickness, contour_matches, 1)

    auto_contour_measures = []

    for result in resultlist:
        organname, scores = result
        # scores[0] APL
        # scores[1] TP volume
        # scores[2] FN volume
        # scores[3] FP volume
        # scores[4] Ref volume
        # scores[5] Test volume
        # scores[6] Hausdorff
        # scores[7] 95% Hausdorff
        # scores[8] Average Distance
        # scores[9] Median Distance
        # scores[10] Reference Centroid
        # scores[11] Test Centroid
        results_structure = {'Organ': organname, 'APL': scores[0], 'TPVol': scores[1], 'FNVol': scores[2],
                             'FPVol': scores[3], 'SEN': scores[1] / scores[4], 'SFP': scores[3] / scores[5],
                             'three_D_DSC': 2 * scores[1] / (scores[4] + scores[5]), 'HD': scores[6],
                             'ninety_five_HD': scores[7], 'AD': scores[8], 'MD': scores[9], 'ref_cent': scores[10],
                             'test_cent': scores[11]}
        auto_contour_measures.append(results_structure)

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


def main():
    # Find measures for two dicom RTSS. First is "ground truth", second is "test"
    print('Scoring Stuff\n')
    # hardcode for now.
    # TODO Add a parse for arguments

    # ground_truth_file = 'C:/Mark/Book/Scoring code/TruthContours.dcm'
    # ground_truth_file = 'C:\\Mark\Book\\Scoring code\\TestCases\\LCTSC-Test-S1-102\\Reference\\IM1.dcm'
    # ground_truth_file = 'C:\\Mark\\Book\\Scoring code\\TestCases\\Artificial contours\\APLtestGT/IM1.dcm'
    ground_truth_file = 'C:\\Mark\\Book\\Scoring code\\TestCases\\Consensus\\' \
                        'Contoured by Research HN Consensus Atlas All\\IM1.DCM'
    # ground_truth_file = 'C:\Mark\Book\Scoring code\Full.DCM'
    # test_file = 'C:/Mark/Book/Scoring code/TestContours.dcm'
    # test_file = 'C:\\Mark\\Book\\Scoring code\\TestCases\\LCTSC-Test-S1-102\\Test\\IM1.dcm'
    # test_file = 'C:\\Mark\\Book\\Scoring code\\TestCases\\Artificial contours\\APLtestTest/IM1.dcm'
    test_file = 'C:\\Mark\\Book\\Scoring code\\TestCases\\Consensus\\DLCWithBrain\\IM1.DCM'
    # test_file = 'C:\Mark\Book\Scoring code\Single.DCM'
    output_file = 'C:/Mark/Book/Scoring code/Results.csv'

    score_case(ground_truth_file, test_file, 0, output_file)


if __name__ == '__main__':
    main()
