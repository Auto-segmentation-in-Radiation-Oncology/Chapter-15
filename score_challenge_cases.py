import score_autocontours
import argparse
import csv
import numpy as np
import time


def main():
    help_text = 'Scoring the test cases for the AAPM Throacic challenge'
    parser = argparse.ArgumentParser(description=help_text)

    help_text = 'Root location of the reference cases with "ground truth"'
    parser.add_argument('Reference_root', type=str, help=help_text)

    help_text = 'Root location of the test cases to score'
    parser.add_argument('Test_root', type=str, help=help_text)

    help_text = 'CSV file in which to store results'
    parser.add_argument('Output_CSV', type=str, help=help_text)

    help_text = 'Set to "True" if running tests for live challenge data, '
    parser.add_argument('-Live_challenge', type=bool, help=help_text, default=False)

    args = parser.parse_args()

    start_time = time.time()

    if not args.Live_challenge:
        case_name = ['LCTSC-Test-S1-101.dcm',
                     'LCTSC-Test-S1-102.dcm',
                     'LCTSC-Test-S1-103.dcm',
                     'LCTSC-Test-S1-104.dcm',
                     'LCTSC-Test-S2-101.dcm',
                     'LCTSC-Test-S2-102.dcm',
                     'LCTSC-Test-S2-103.dcm',
                     'LCTSC-Test-S2-104.dcm',
                     'LCTSC-Test-S3-101.dcm',
                     'LCTSC-Test-S3-102.dcm',
                     'LCTSC-Test-S3-103.dcm',
                     'LCTSC-Test-S3-104.dcm']
    else:
        case_name = ['LCTSC-Test-S1-201.dcm',
                     'LCTSC-Test-S1-202.dcm',
                     'LCTSC-Test-S1-203.dcm',
                     'LCTSC-Test-S1-204.dcm',
                     'LCTSC-Test-S2-201.dcm',
                     'LCTSC-Test-S2-202.dcm',
                     'LCTSC-Test-S2-203.dcm',
                     'LCTSC-Test-S2-204.dcm',
                     'LCTSC-Test-S3-201.dcm',
                     'LCTSC-Test-S3-202.dcm',
                     'LCTSC-Test-S3-203.dcm',
                     'LCTSC-Test-S3-204.dcm']

    case_score = {}

    for case in case_name:
        test_data_file = '{:s}{:s}'.format(args.Test_root, case)
        ref_data_file = '{:s}{:s}'.format(args.Reference_root, case)
        scoresfound = score_autocontours.score_case(ref_data_file, test_data_file)
        case_score[case] = scoresfound

    elapsed_time = time.time() - start_time

    print('Calculation time required: ', elapsed_time)

    print('Writing results to: ', args.Output_CSV)
    with open(args.Output_CSV, mode='w', newline='\n', encoding='utf-8') as out_file:
        result_writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        result_writer.writerow(['Segmentation Name', 'Lung_R_DICE', 'Lung_R_HAUSDORFF', 'Lung_R_AVERAGE_DISTANCE',
                                'Esophagus_DICE', 'Esophagus_HAUSDORFF', 'Esophagus_AVERAGE_DISTANCE',
                                'Lung_L_DICE', 'Lung_L_HAUSDORFF', 'Lung_L_AVERAGE_DISTANCE',
                                'Heart_DICE', 'Heart_HAUSDORFF', 'Heart_AVERAGE_DISTANCE',
                                'SpinalCord_DICE', 'SpinalCord_HAUSDORFF', 'SpinalCord_AVERAGE_DISTANCE'])
        for casename, result in case_score.items():
            outputdata = np.zeros(15)
            for results_structure in result:
                if results_structure['Organ'] == 'Lung_R':
                    outputdata[0] = results_structure['three_D_DSC']
                    outputdata[1] = results_structure['ninety_five_HD']
                    outputdata[2] = results_structure['AD']
                elif results_structure['Organ'] == 'Esophagus':
                    outputdata[3] = results_structure['three_D_DSC']
                    outputdata[4] = results_structure['ninety_five_HD']
                    outputdata[5] = results_structure['AD']
                elif results_structure['Organ'] == 'Lung_L':
                    outputdata[6] = results_structure['three_D_DSC']
                    outputdata[7] = results_structure['ninety_five_HD']
                    outputdata[8] = results_structure['AD']
                elif results_structure['Organ'] == 'Heart':
                    outputdata[9] = results_structure['three_D_DSC']
                    outputdata[10] = results_structure['ninety_five_HD']
                    outputdata[11] = results_structure['AD']
                elif results_structure['Organ'] == 'SpinalCord':
                    outputdata[12] = results_structure['three_D_DSC']
                    outputdata[13] = results_structure['ninety_five_HD']
                    outputdata[14] = results_structure['AD']

            result_writer.writerow([casename, outputdata[0], outputdata[1], outputdata[2], outputdata[3], outputdata[4],
                                    outputdata[5], outputdata[6], outputdata[7], outputdata[8], outputdata[9],
                                    outputdata[10], outputdata[11], outputdata[12], outputdata[13], outputdata[14]])


if __name__ == '__main__':
    main()
