import numpy as np
import cPickle

from utils import determine_label

def compute_markov_table():
    #samples = open('/nthome/maugust/thesis/samples_int_ordered.txt')
    samples = open('./samples_int_ordered.txt')
    #labels = open('/nthome/maugust/thesis/labels_int_ordered.txt')
    labels = open('./labels_int_ordered.txt')
    #annotations = open('/nthome/maugust/thesis/annotations_int_ordered.txt')
    annotations = open('./annotations_int_ordered.txt')

    bad_samples = []
    real_labels = []
    qpoint_lists = []
    label_list = []
    annotation_list = []
    label_count = np.zeros((1,13))

    for data in samples:
        qpoint_lists = data.split(';')
    for data in labels:
        label_list = data.split(';')
    for data in annotations:
        annotation_list = data.split(';')

    print 'found %i qpoint lists.' % len(qpoint_lists)
    print 'found %i labels.' % len(label_list)
    print 'found %i annotations.' % len(annotation_list)

    for list_ind in np.arange(len(qpoint_lists)):
        bad = False

        ################# PROCESS THE LABELS
        if annotation_list[list_ind][0:2] != 'vo' and annotation_list[list_ind][0:2] != 'fl' and annotation_list[list_ind][0:2] != 'mi' and annotation_list[list_ind][0:2] != 'ja':
            real_labels.append(0)
            label_count[0][0] += 1
        else:
            position = label_list[list_ind].split(',')
            if float(position[0]) == -2000 or float(position[0]) == -1000:
                real_labels.append(-1)
                bad = True
            else:
                lab = determine_label((float(position[0]),float(position[1]),float(position[2])))
                real_labels.append(lab)
                label_count[0][lab] += 1

        ################# PROCESS THE Q-POINTS
        qpoint_lists[list_ind] = qpoint_lists[list_ind].split(':')
        for point_ind in np.arange(len(qpoint_lists[list_ind])):
            qpoint_lists[list_ind][point_ind] = qpoint_lists[list_ind][point_ind].split(',')
            if len(qpoint_lists[list_ind][point_ind]) != 7:
                bad = True

        if bad:
            bad_samples.append(list_ind)

    print 'need to remove %i bad samples.' %len(bad_samples)
    ################# REMOVE BAD SAMPLES
    ind = 0
    for bad_ind in bad_samples:
        real_ind = bad_ind - ind
        qpoint_lists.pop(real_ind)
        real_labels.pop(real_ind)
        annotation_list.pop(real_ind)
        ind += 1

    print str(len(qpoint_lists)) + ' samples remain after purging.'
    print str(len(real_labels)) + ' labels remain after purging.'
    print str(len(annotation_list)) + ' annotations remain after purging.'
    print 'percentages of the labels are %s' %str(label_count/len(qpoint_lists))
    samples.close()
    labels.close()
    annotations.close()


    ################################################### COMPUTE THE TABLE ##############################################

    prior_table = np.zeros((13, 13, 13), dtype=np.float32)
    stat_table = np.zeros((13, 13, 13), dtype=np.float32)
    posterior_table = np.zeros((13,13,13), dtype=np.float32)

    prior_table[0][0][0] = 1./11
    prior_table[0][0][1] = 1./11
    prior_table[0][0][2] = 1./11
    prior_table[0][0][3] = 1./11
    prior_table[0][0][4] = 1./11
    prior_table[0][0][5] = 1./11
    prior_table[0][0][6] = 0.
    prior_table[0][0][7] = 0.
    prior_table[0][0][8] = 1./11
    prior_table[0][0][9] = 1./11
    prior_table[0][0][10] = 1./11
    prior_table[0][0][11] = 1./11
    prior_table[0][0][12] = 1./11

    prior_table[1][0][0] = 1./11
    prior_table[1][0][1] = 1./11
    prior_table[1][0][2] = 1./11
    prior_table[1][0][3] = 1./11
    prior_table[1][0][4] = 1./11
    prior_table[1][0][5] = 1./11
    prior_table[1][0][6] = 0.
    prior_table[1][0][7] = 0.
    prior_table[1][0][8] = 1./11
    prior_table[1][0][9] = 1./11
    prior_table[1][0][10] = 1./11
    prior_table[1][0][11] = 1./11
    prior_table[1][0][12] = 1./11

    prior_table[2][0][0] = 1./11
    prior_table[2][0][1] = 1./11
    prior_table[2][0][2] = 1./11
    prior_table[2][0][3] = 1./11
    prior_table[2][0][4] = 1./11
    prior_table[2][0][5] = 1./11
    prior_table[2][0][6] = 0.
    prior_table[2][0][7] = 0.
    prior_table[2][0][8] = 1./11
    prior_table[2][0][9] = 1./11
    prior_table[2][0][10] = 1./11
    prior_table[2][0][11] = 1./11
    prior_table[2][0][12] = 1./11

    prior_table[3][0][0] = 1./11
    prior_table[3][0][1] = 1./11
    prior_table[3][0][2] = 1./11
    prior_table[3][0][3] = 1./11
    prior_table[3][0][4] = 1./11
    prior_table[3][0][5] = 1./11
    prior_table[3][0][6] = 0.
    prior_table[3][0][7] = 0.
    prior_table[3][0][8] = 1./11
    prior_table[3][0][9] = 1./11
    prior_table[3][0][10] = 1./11
    prior_table[3][0][11] = 1./11
    prior_table[3][0][12] = 1./11

    prior_table[4][0][0] = 1./11
    prior_table[4][0][1] = 1./11
    prior_table[4][0][2] = 1./11
    prior_table[4][0][3] = 1./11
    prior_table[4][0][4] = 1./11
    prior_table[4][0][5] = 1./11
    prior_table[4][0][6] = 0.
    prior_table[4][0][7] = 0.
    prior_table[4][0][8] = 1./11
    prior_table[4][0][9] = 1./11
    prior_table[4][0][10] = 1./11
    prior_table[4][0][11] = 1./11
    prior_table[4][0][12] = 1./11

    prior_table[5][0][0] = 1./11
    prior_table[5][0][1] = 1./11
    prior_table[5][0][2] = 1./11
    prior_table[5][0][3] = 1./11
    prior_table[5][0][4] = 1./11
    prior_table[5][0][5] = 1./11
    prior_table[5][0][6] = 0.
    prior_table[5][0][7] = 0.
    prior_table[5][0][8] = 1./11
    prior_table[5][0][9] = 1./11
    prior_table[5][0][10] = 1./11
    prior_table[5][0][11] = 1./11
    prior_table[5][0][12] = 1./11

    prior_table[6][0][0] = 1./11
    prior_table[6][0][1] = 1./11
    prior_table[6][0][2] = 1./11
    prior_table[6][0][3] = 1./11
    prior_table[6][0][4] = 1./11
    prior_table[6][0][5] = 1./11
    prior_table[6][0][6] = 0.
    prior_table[6][0][7] = 0.
    prior_table[6][0][8] = 1./11
    prior_table[6][0][9] = 1./11
    prior_table[6][0][10] = 1./11
    prior_table[6][0][11] = 1./11
    prior_table[6][0][12] = 1./11

    prior_table[7][0][0] = 1./11
    prior_table[7][0][1] = 1./11
    prior_table[7][0][2] = 1./11
    prior_table[7][0][3] = 1./11
    prior_table[7][0][4] = 1./11
    prior_table[7][0][5] = 1./11
    prior_table[7][0][6] = 0.
    prior_table[7][0][7] = 0.
    prior_table[7][0][8] = 1./11
    prior_table[7][0][9] = 1./11
    prior_table[7][0][10] = 1./11
    prior_table[7][0][11] = 1./11
    prior_table[7][0][12] = 1./11

    prior_table[8][0][0] = 1./11
    prior_table[8][0][1] = 1./11
    prior_table[8][0][2] = 1./11
    prior_table[8][0][3] = 1./11
    prior_table[8][0][4] = 1./11
    prior_table[8][0][5] = 1./11
    prior_table[8][0][6] = 0.
    prior_table[8][0][7] = 0.
    prior_table[8][0][8] = 1./11
    prior_table[8][0][9] = 1./11
    prior_table[8][0][10] = 1./11
    prior_table[8][0][11] = 1./11
    prior_table[8][0][12] = 1./11

    prior_table[9][0][0] = 1./11
    prior_table[9][0][1] = 1./11
    prior_table[9][0][2] = 1./11
    prior_table[9][0][3] = 1./11
    prior_table[9][0][4] = 1./11
    prior_table[9][0][5] = 1./11
    prior_table[9][0][6] = 0.
    prior_table[9][0][7] = 0.
    prior_table[9][0][8] = 1./11
    prior_table[9][0][9] = 1./11
    prior_table[9][0][10] = 1./11
    prior_table[9][0][11] = 1./11
    prior_table[9][0][12] = 1./11

    prior_table[10][0][0] = 1./11
    prior_table[10][0][1] = 1./11
    prior_table[10][0][2] = 1./11
    prior_table[10][0][3] = 1./11
    prior_table[10][0][4] = 1./11
    prior_table[10][0][5] = 1./11
    prior_table[10][0][6] = 0.
    prior_table[10][0][7] = 0.
    prior_table[10][0][8] = 1./11
    prior_table[10][0][9] = 1./11
    prior_table[10][0][10] = 1./11
    prior_table[10][0][11] = 1./11
    prior_table[10][0][12] = 1./11

    prior_table[11][0][0] = 1./11
    prior_table[11][0][1] = 1./11
    prior_table[11][0][2] = 1./11
    prior_table[11][0][3] = 1./11
    prior_table[11][0][4] = 1./11
    prior_table[11][0][5] = 1./11
    prior_table[11][0][6] = 0.
    prior_table[11][0][7] = 0.
    prior_table[11][0][8] = 1./11
    prior_table[11][0][9] = 1./11
    prior_table[11][0][10] = 1./11
    prior_table[11][0][11] = 1./11
    prior_table[11][0][12] = 1./11

    prior_table[12][0][0] = 1./11
    prior_table[12][0][1] = 1./11
    prior_table[12][0][2] = 1./11
    prior_table[12][0][3] = 1./11
    prior_table[12][0][4] = 1./11
    prior_table[12][0][5] = 1./11
    prior_table[12][0][6] = 0.
    prior_table[12][0][7] = 0.
    prior_table[12][0][8] = 1./11
    prior_table[12][0][9] = 1./11
    prior_table[12][0][10] = 1./11
    prior_table[12][0][11] = 1./11
    prior_table[12][0][12] = 1./11

    ####################################################################################

    prior_table[0][1][0] = 1./5
    prior_table[0][1][1] = 1./5
    prior_table[0][1][2] = 1./5
    prior_table[0][1][3] = 0.
    prior_table[0][1][4] = 0.
    prior_table[0][1][5] = 1./5
    prior_table[0][1][6] = 1./5
    prior_table[0][1][7] = 0.
    prior_table[0][1][8] = 0.
    prior_table[0][1][9] = 0.
    prior_table[0][1][10] = 0.
    prior_table[0][1][11] = 0.
    prior_table[0][1][12] = 0.

    prior_table[1][1][0] = 1./5
    prior_table[1][1][1] = 1./5
    prior_table[1][1][2] = 1./5
    prior_table[1][1][3] = 0.
    prior_table[1][1][4] = 0.
    prior_table[1][1][5] = 1./5
    prior_table[1][1][6] = 1./5
    prior_table[1][1][7] = 0.
    prior_table[1][1][8] = 0.
    prior_table[1][1][9] = 0.
    prior_table[1][1][10] = 0.
    prior_table[1][1][11] = 0.
    prior_table[1][1][12] = 0.

    prior_table[2][1][0] = 1./5
    prior_table[2][1][1] = 1./5
    prior_table[2][1][2] = 1./5
    prior_table[2][1][3] = 0.
    prior_table[2][1][4] = 0.
    prior_table[2][1][5] = 1./5
    prior_table[2][1][6] = 1./5
    prior_table[2][1][7] = 0.
    prior_table[2][1][8] = 0.
    prior_table[2][1][9] = 0.
    prior_table[2][1][10] = 0.
    prior_table[2][1][11] = 0.
    prior_table[2][1][12] = 0.

    prior_table[3][1][0] = 1./5
    prior_table[3][1][1] = 1./5
    prior_table[3][1][2] = 1./5
    prior_table[3][1][3] = 0.
    prior_table[3][1][4] = 0.
    prior_table[3][1][5] = 1./5
    prior_table[3][1][6] = 1./5
    prior_table[3][1][7] = 0.
    prior_table[3][1][8] = 0.
    prior_table[3][1][9] = 0.
    prior_table[3][1][10] = 0.
    prior_table[3][1][11] = 0.
    prior_table[3][1][12] = 0.

    prior_table[4][1][0] = 1./5
    prior_table[4][1][1] = 1./5
    prior_table[4][1][2] = 1./5
    prior_table[4][1][3] = 0.
    prior_table[4][1][4] = 0.
    prior_table[4][1][5] = 1./5
    prior_table[4][1][6] = 1./5
    prior_table[4][1][7] = 0.
    prior_table[4][1][8] = 0.
    prior_table[4][1][9] = 0.
    prior_table[4][1][10] = 0.
    prior_table[4][1][11] = 0.
    prior_table[4][1][12] = 0.

    prior_table[5][1][0] = 1./5
    prior_table[5][1][1] = 1./5
    prior_table[5][1][2] = 1./5
    prior_table[5][1][3] = 0.
    prior_table[5][1][4] = 0.
    prior_table[5][1][5] = 1./5
    prior_table[5][1][6] = 1./5
    prior_table[5][1][7] = 0.
    prior_table[5][1][8] = 0.
    prior_table[5][1][9] = 0.
    prior_table[5][1][10] = 0.
    prior_table[5][1][11] = 0.
    prior_table[5][1][12] = 0.

    prior_table[6][1][0] = 1./5
    prior_table[6][1][1] = 1./5
    prior_table[6][1][2] = 1./5
    prior_table[6][1][3] = 0.
    prior_table[6][1][4] = 0.
    prior_table[6][1][5] = 1./5
    prior_table[6][1][6] = 1./5
    prior_table[6][1][7] = 0.
    prior_table[6][1][8] = 0.
    prior_table[6][1][9] = 0.
    prior_table[6][1][10] = 0.
    prior_table[6][1][11] = 0.
    prior_table[6][1][12] = 0.

    prior_table[7][1][0] = 1./5
    prior_table[7][1][1] = 1./5
    prior_table[7][1][2] = 1./5
    prior_table[7][1][3] = 0.
    prior_table[7][1][4] = 0.
    prior_table[7][1][5] = 1./5
    prior_table[7][1][6] = 1./5
    prior_table[7][1][7] = 0.
    prior_table[7][1][8] = 0.
    prior_table[7][1][9] = 0.
    prior_table[7][1][10] = 0.
    prior_table[7][1][11] = 0.
    prior_table[7][1][12] = 0.

    prior_table[8][1][0] = 1./5
    prior_table[8][1][1] = 1./5
    prior_table[8][1][2] = 1./5
    prior_table[8][1][3] = 0.
    prior_table[8][1][4] = 0.
    prior_table[8][1][5] = 1./5
    prior_table[8][1][6] = 1./5
    prior_table[8][1][7] = 0.
    prior_table[8][1][8] = 0.
    prior_table[8][1][9] = 0.
    prior_table[8][1][10] = 0.
    prior_table[8][1][11] = 0.
    prior_table[8][1][12] = 0.

    prior_table[9][1][0] = 1./5
    prior_table[9][1][1] = 1./5
    prior_table[9][1][2] = 1./5
    prior_table[9][1][3] = 0.
    prior_table[9][1][4] = 0.
    prior_table[9][1][5] = 1./5
    prior_table[9][1][6] = 1./5
    prior_table[9][1][7] = 0.
    prior_table[9][1][8] = 0.
    prior_table[9][1][9] = 0.
    prior_table[9][1][10] = 0.
    prior_table[9][1][11] = 0.
    prior_table[9][1][12] = 0.

    prior_table[10][1][0] = 1./5
    prior_table[10][1][1] = 1./5
    prior_table[10][1][2] = 1./5
    prior_table[10][1][3] = 0.
    prior_table[10][1][4] = 0.
    prior_table[10][1][5] = 1./5
    prior_table[10][1][6] = 1./5
    prior_table[10][1][7] = 0.
    prior_table[10][1][8] = 0.
    prior_table[10][1][9] = 0.
    prior_table[10][1][10] = 0.
    prior_table[10][1][11] = 0.
    prior_table[10][1][12] = 0.

    prior_table[11][1][0] = 1./5
    prior_table[11][1][1] = 1./5
    prior_table[11][1][2] = 1./5
    prior_table[11][1][3] = 0.
    prior_table[11][1][4] = 0.
    prior_table[11][1][5] = 1./5
    prior_table[11][1][6] = 1./5
    prior_table[11][1][7] = 0.
    prior_table[11][1][8] = 0.
    prior_table[11][1][9] = 0.
    prior_table[11][1][10] = 0.
    prior_table[11][1][11] = 0.
    prior_table[11][1][12] = 0.

    prior_table[12][1][0] = 1./5
    prior_table[12][1][1] = 1./5
    prior_table[12][1][2] = 1./5
    prior_table[12][1][3] = 0.
    prior_table[12][1][4] = 0.
    prior_table[12][1][5] = 1./5
    prior_table[12][1][6] = 1./5
    prior_table[12][1][7] = 0.
    prior_table[12][1][8] = 0.
    prior_table[12][1][9] = 0.
    prior_table[12][1][10] = 0.
    prior_table[12][1][11] = 0.
    prior_table[12][1][12] = 0.

    #################################################################

    prior_table[0][2][0] = 1./7
    prior_table[0][2][1] = 1./7
    prior_table[0][2][2] = 1./7
    prior_table[0][2][3] = 1./7
    prior_table[0][2][4] = 0.
    prior_table[0][2][5] = 1./7
    prior_table[0][2][6] = 1./7
    prior_table[0][2][7] = 1./7
    prior_table[0][2][8] = 0.
    prior_table[0][2][9] = 0.
    prior_table[0][2][10] = 0.
    prior_table[0][2][11] = 0.
    prior_table[0][2][12] = 0.

    prior_table[1][2][0] = 1./7
    prior_table[1][2][1] = 1./7
    prior_table[1][2][2] = 1./7
    prior_table[1][2][3] = 1./7
    prior_table[1][2][4] = 0.
    prior_table[1][2][5] = 1./7
    prior_table[1][2][6] = 1./7
    prior_table[1][2][7] = 1./7
    prior_table[1][2][8] = 0.
    prior_table[1][2][9] = 0.
    prior_table[1][2][10] = 0.
    prior_table[1][2][11] = 0.
    prior_table[1][2][12] = 0.

    prior_table[2][2][0] = 1./7
    prior_table[2][2][1] = 1./7
    prior_table[2][2][2] = 1./7
    prior_table[2][2][3] = 1./7
    prior_table[2][2][4] = 0.
    prior_table[2][2][5] = 1./7
    prior_table[2][2][6] = 1./7
    prior_table[2][2][7] = 1./7
    prior_table[2][2][8] = 0.
    prior_table[2][2][9] = 0.
    prior_table[2][2][10] = 0.
    prior_table[2][2][11] = 0.
    prior_table[2][2][12] = 0.

    prior_table[3][2][0] = 1./7
    prior_table[3][2][1] = 1./7
    prior_table[3][2][2] = 1./7
    prior_table[3][2][3] = 1./7
    prior_table[3][2][4] = 0.
    prior_table[3][2][5] = 1./7
    prior_table[3][2][6] = 1./7
    prior_table[3][2][7] = 1./7
    prior_table[3][2][8] = 0.
    prior_table[3][2][9] = 0.
    prior_table[3][2][10] = 0.
    prior_table[3][2][11] = 0.
    prior_table[3][2][12] = 0.

    prior_table[4][2][0] = 1./7
    prior_table[4][2][1] = 1./7
    prior_table[4][2][2] = 1./7
    prior_table[4][2][3] = 1./7
    prior_table[4][2][4] = 0.
    prior_table[4][2][5] = 1./7
    prior_table[4][2][6] = 1./7
    prior_table[4][2][7] = 1./7
    prior_table[4][2][8] = 0.
    prior_table[4][2][9] = 0.
    prior_table[4][2][10] = 0.
    prior_table[4][2][11] = 0.
    prior_table[4][2][12] = 0.

    prior_table[5][2][0] = 1./7
    prior_table[5][2][1] = 1./7
    prior_table[5][2][2] = 1./7
    prior_table[5][2][3] = 1./7
    prior_table[5][2][4] = 0.
    prior_table[5][2][5] = 1./7
    prior_table[5][2][6] = 1./7
    prior_table[5][2][7] = 1./7
    prior_table[5][2][8] = 0.
    prior_table[5][2][9] = 0.
    prior_table[5][2][10] = 0.
    prior_table[5][2][11] = 0.
    prior_table[5][2][12] = 0.

    prior_table[6][2][0] = 1./7
    prior_table[6][2][1] = 1./7
    prior_table[6][2][2] = 1./7
    prior_table[6][2][3] = 1./7
    prior_table[6][2][4] = 0.
    prior_table[6][2][5] = 1./7
    prior_table[6][2][6] = 1./7
    prior_table[6][2][7] = 1./7
    prior_table[6][2][8] = 0.
    prior_table[6][2][9] = 0.
    prior_table[6][2][10] = 0.
    prior_table[6][2][11] = 0.
    prior_table[6][2][12] = 0.

    prior_table[7][2][0] = 1./7
    prior_table[7][2][1] = 1./7
    prior_table[7][2][2] = 1./7
    prior_table[7][2][3] = 1./7
    prior_table[7][2][4] = 0.
    prior_table[7][2][5] = 1./7
    prior_table[7][2][6] = 1./7
    prior_table[7][2][7] = 1./7
    prior_table[7][2][8] = 0.
    prior_table[7][2][9] = 0.
    prior_table[7][2][10] = 0.
    prior_table[7][2][11] = 0.
    prior_table[7][2][12] = 0.

    prior_table[8][2][0] = 1./7
    prior_table[8][2][1] = 1./7
    prior_table[8][2][2] = 1./7
    prior_table[8][2][3] = 1./7
    prior_table[8][2][4] = 0.
    prior_table[8][2][5] = 1./7
    prior_table[8][2][6] = 1./7
    prior_table[8][2][7] = 1./7
    prior_table[8][2][8] = 0.
    prior_table[8][2][9] = 0.
    prior_table[8][2][10] = 0.
    prior_table[8][2][11] = 0.
    prior_table[8][2][12] = 0.

    prior_table[9][2][0] = 1./7
    prior_table[9][2][1] = 1./7
    prior_table[9][2][2] = 1./7
    prior_table[9][2][3] = 1./7
    prior_table[9][2][4] = 0.
    prior_table[9][2][5] = 1./7
    prior_table[9][2][6] = 1./7
    prior_table[9][2][7] = 1./7
    prior_table[9][2][8] = 0.
    prior_table[9][2][9] = 0.
    prior_table[9][2][10] = 0.
    prior_table[9][2][11] = 0.
    prior_table[9][2][12] = 0.

    prior_table[10][2][0] = 1./7
    prior_table[10][2][1] = 1./7
    prior_table[10][2][2] = 1./7
    prior_table[10][2][3] = 1./7
    prior_table[10][2][4] = 0.
    prior_table[10][2][5] = 1./7
    prior_table[10][2][6] = 1./7
    prior_table[10][2][7] = 1./7
    prior_table[10][2][8] = 0.
    prior_table[10][2][9] = 0.
    prior_table[10][2][10] = 0.
    prior_table[10][2][11] = 0.
    prior_table[10][2][12] = 0.

    prior_table[11][2][0] = 1./7
    prior_table[11][2][1] = 1./7
    prior_table[11][2][2] = 1./7
    prior_table[11][2][3] = 1./7
    prior_table[11][2][4] = 0.
    prior_table[11][2][5] = 1./7
    prior_table[11][2][6] = 1./7
    prior_table[11][2][7] = 1./7
    prior_table[11][2][8] = 0.
    prior_table[11][2][9] = 0.
    prior_table[11][2][10] = 0.
    prior_table[11][2][11] = 0.
    prior_table[11][2][12] = 0.

    prior_table[12][2][0] = 1./7
    prior_table[12][2][1] = 1./7
    prior_table[12][2][2] = 1./7
    prior_table[12][2][3] = 1./7
    prior_table[12][2][4] = 0.
    prior_table[12][2][5] = 1./7
    prior_table[12][2][6] = 1./7
    prior_table[12][2][7] = 1./7
    prior_table[12][2][8] = 0.
    prior_table[12][2][9] = 0.
    prior_table[12][2][10] = 0.
    prior_table[12][2][11] = 0.
    prior_table[12][2][12] = 0.

    #################################################################

    prior_table[0][3][0] = 1./7
    prior_table[0][3][1] = 0.
    prior_table[0][3][2] = 1./7
    prior_table[0][3][3] = 1./7
    prior_table[0][3][4] = 1./7
    prior_table[0][3][5] = 0.
    prior_table[0][3][6] = 1./7
    prior_table[0][3][7] = 1./7
    prior_table[0][3][8] = 1./7
    prior_table[0][3][9] = 0.
    prior_table[0][3][10] = 0.
    prior_table[0][3][11] = 0.
    prior_table[0][3][12] = 0.

    prior_table[1][3][0] = 1./7
    prior_table[1][3][1] = 0.
    prior_table[1][3][2] = 1./7
    prior_table[1][3][3] = 1./7
    prior_table[1][3][4] = 1./7
    prior_table[1][3][5] = 0.
    prior_table[1][3][6] = 1./7
    prior_table[1][3][7] = 1./7
    prior_table[1][3][8] = 1./7
    prior_table[1][3][9] = 0.
    prior_table[1][3][10] = 0.
    prior_table[1][3][11] = 0.
    prior_table[1][3][12] = 0.

    prior_table[2][3][0] = 1./7
    prior_table[2][3][1] = 0.
    prior_table[2][3][2] = 1./7
    prior_table[2][3][3] = 1./7
    prior_table[2][3][4] = 1./7
    prior_table[2][3][5] = 0.
    prior_table[2][3][6] = 1./7
    prior_table[2][3][7] = 1./7
    prior_table[2][3][8] = 1./7
    prior_table[2][3][9] = 0.
    prior_table[2][3][10] = 0.
    prior_table[2][3][11] = 0.
    prior_table[2][3][12] = 0.

    prior_table[3][3][0] = 1./7
    prior_table[3][3][1] = 0.
    prior_table[3][3][2] = 1./7
    prior_table[3][3][3] = 1./7
    prior_table[3][3][4] = 1./7
    prior_table[3][3][5] = 0.
    prior_table[3][3][6] = 1./7
    prior_table[3][3][7] = 1./7
    prior_table[3][3][8] = 1./7
    prior_table[3][3][9] = 0.
    prior_table[3][3][10] = 0.
    prior_table[3][3][11] = 0.
    prior_table[3][3][12] = 0.

    prior_table[4][3][0] = 1./7
    prior_table[4][3][1] = 0.
    prior_table[4][3][2] = 1./7
    prior_table[4][3][3] = 1./7
    prior_table[4][3][4] = 1./7
    prior_table[4][3][5] = 0.
    prior_table[4][3][6] = 1./7
    prior_table[4][3][7] = 1./7
    prior_table[4][3][8] = 1./7
    prior_table[4][3][9] = 0.
    prior_table[4][3][10] = 0.
    prior_table[4][3][11] = 0.
    prior_table[4][3][12] = 0.

    prior_table[5][3][0] = 1./7
    prior_table[5][3][1] = 0.
    prior_table[5][3][2] = 1./7
    prior_table[5][3][3] = 1./7
    prior_table[5][3][4] = 1./7
    prior_table[5][3][5] = 0.
    prior_table[5][3][6] = 1./7
    prior_table[5][3][7] = 1./7
    prior_table[5][3][8] = 1./7
    prior_table[5][3][9] = 0.
    prior_table[5][3][10] = 0.
    prior_table[5][3][11] = 0.
    prior_table[5][3][12] = 0.

    prior_table[6][3][0] = 1./7
    prior_table[6][3][1] = 0.
    prior_table[6][3][2] = 1./7
    prior_table[6][3][3] = 1./7
    prior_table[6][3][4] = 1./7
    prior_table[6][3][5] = 0.
    prior_table[6][3][6] = 1./7
    prior_table[6][3][7] = 1./7
    prior_table[6][3][8] = 1./7
    prior_table[6][3][9] = 0.
    prior_table[6][3][10] = 0.
    prior_table[6][3][11] = 0.
    prior_table[6][3][12] = 0.

    prior_table[7][3][0] = 1./7
    prior_table[7][3][1] = 0.
    prior_table[7][3][2] = 1./7
    prior_table[7][3][3] = 1./7
    prior_table[7][3][4] = 1./7
    prior_table[7][3][5] = 0.
    prior_table[7][3][6] = 1./7
    prior_table[7][3][7] = 1./7
    prior_table[7][3][8] = 1./7
    prior_table[7][3][9] = 0.
    prior_table[7][3][10] = 0.
    prior_table[7][3][11] = 0.
    prior_table[7][3][12] = 0.

    prior_table[8][3][0] = 1./7
    prior_table[8][3][1] = 0.
    prior_table[8][3][2] = 1./7
    prior_table[8][3][3] = 1./7
    prior_table[8][3][4] = 1./7
    prior_table[8][3][5] = 0.
    prior_table[8][3][6] = 1./7
    prior_table[8][3][7] = 1./7
    prior_table[8][3][8] = 1./7
    prior_table[8][3][9] = 0.
    prior_table[8][3][10] = 0.
    prior_table[8][3][11] = 0.
    prior_table[8][3][12] = 0.

    prior_table[9][3][0] = 1./7
    prior_table[9][3][1] = 0.
    prior_table[9][3][2] = 1./7
    prior_table[9][3][3] = 1./7
    prior_table[9][3][4] = 1./7
    prior_table[9][3][5] = 0.
    prior_table[9][3][6] = 1./7
    prior_table[9][3][7] = 1./7
    prior_table[9][3][8] = 1./7
    prior_table[9][3][9] = 0.
    prior_table[9][3][10] = 0.
    prior_table[9][3][11] = 0.
    prior_table[9][3][12] = 0.

    prior_table[10][3][0] = 1./7
    prior_table[10][3][1] = 0.
    prior_table[10][3][2] = 1./7
    prior_table[10][3][3] = 1./7
    prior_table[10][3][4] = 1./7
    prior_table[10][3][5] = 0.
    prior_table[10][3][6] = 1./7
    prior_table[10][3][7] = 1./7
    prior_table[10][3][8] = 1./7
    prior_table[10][3][9] = 0.
    prior_table[10][3][10] = 0.
    prior_table[10][3][11] = 0.
    prior_table[10][3][12] = 0.

    prior_table[11][3][0] = 1./7
    prior_table[11][3][1] = 0.
    prior_table[11][3][2] = 1./7
    prior_table[11][3][3] = 1./7
    prior_table[11][3][4] = 1./7
    prior_table[11][3][5] = 0.
    prior_table[11][3][6] = 1./7
    prior_table[11][3][7] = 1./7
    prior_table[11][3][8] = 1./7
    prior_table[11][3][9] = 0.
    prior_table[11][3][10] = 0.
    prior_table[11][3][11] = 0.
    prior_table[11][3][12] = 0.

    prior_table[12][3][0] = 1./7
    prior_table[12][3][1] = 0.
    prior_table[12][3][2] = 1./7
    prior_table[12][3][3] = 1./7
    prior_table[12][3][4] = 1./7
    prior_table[12][3][5] = 0.
    prior_table[12][3][6] = 1./7
    prior_table[12][3][7] = 1./7
    prior_table[12][3][8] = 1./7
    prior_table[12][3][9] = 0.
    prior_table[12][3][10] = 0.
    prior_table[12][3][11] = 0.
    prior_table[12][3][12] = 0.

    ##################################################################

    prior_table[0][4][0] = 1./5
    prior_table[0][4][1] = 0.
    prior_table[0][4][2] = 0.
    prior_table[0][4][3] = 1./5
    prior_table[0][4][4] = 1./5
    prior_table[0][4][5] = 0.
    prior_table[0][4][6] = 0.
    prior_table[0][4][7] = 1./5
    prior_table[0][4][8] = 1./5
    prior_table[0][4][9] = 0.
    prior_table[0][4][10] = 0.
    prior_table[0][4][11] = 0.
    prior_table[0][4][12] = 0.

    prior_table[1][4][0] = 1./5
    prior_table[1][4][1] = 0.
    prior_table[1][4][2] = 0.
    prior_table[1][4][3] = 1./5
    prior_table[1][4][4] = 1./5
    prior_table[1][4][5] = 0.
    prior_table[1][4][6] = 0.
    prior_table[1][4][7] = 1./5
    prior_table[1][4][8] = 1./5
    prior_table[1][4][9] = 0.
    prior_table[1][4][10] = 0.
    prior_table[1][4][11] = 0.
    prior_table[1][4][12] = 0.

    prior_table[2][4][0] = 1./5
    prior_table[2][4][1] = 0.
    prior_table[2][4][2] = 0.
    prior_table[2][4][3] = 1./5
    prior_table[2][4][4] = 1./5
    prior_table[2][4][5] = 0.
    prior_table[2][4][6] = 0.
    prior_table[2][4][7] = 1./5
    prior_table[2][4][8] = 1./5
    prior_table[2][4][9] = 0.
    prior_table[2][4][10] = 0.
    prior_table[2][4][11] = 0.
    prior_table[2][4][12] = 0.

    prior_table[3][4][0] = 1./5
    prior_table[3][4][1] = 0.
    prior_table[3][4][2] = 0.
    prior_table[3][4][3] = 1./5
    prior_table[3][4][4] = 1./5
    prior_table[3][4][5] = 0.
    prior_table[3][4][6] = 0.
    prior_table[3][4][7] = 1./5
    prior_table[3][4][8] = 1./5
    prior_table[3][4][9] = 0.
    prior_table[3][4][10] = 0.
    prior_table[3][4][11] = 0.
    prior_table[3][4][12] = 0.

    prior_table[4][4][0] = 1./5
    prior_table[4][4][1] = 0.
    prior_table[4][4][2] = 0.
    prior_table[4][4][3] = 1./5
    prior_table[4][4][4] = 1./5
    prior_table[4][4][5] = 0.
    prior_table[4][4][6] = 0.
    prior_table[4][4][7] = 1./5
    prior_table[4][4][8] = 1./5
    prior_table[4][4][9] = 0.
    prior_table[4][4][10] = 0.
    prior_table[4][4][11] = 0.
    prior_table[4][4][12] = 0.

    prior_table[5][4][0] = 1./5
    prior_table[5][4][1] = 0.
    prior_table[5][4][2] = 0.
    prior_table[5][4][3] = 1./5
    prior_table[5][4][4] = 1./5
    prior_table[5][4][5] = 0.
    prior_table[5][4][6] = 0.
    prior_table[5][4][7] = 1./5
    prior_table[5][4][8] = 1./5
    prior_table[5][4][9] = 0.
    prior_table[5][4][10] = 0.
    prior_table[5][4][11] = 0.
    prior_table[5][4][12] = 0.

    prior_table[6][4][0] = 1./5
    prior_table[6][4][1] = 0.
    prior_table[6][4][2] = 0.
    prior_table[6][4][3] = 1./5
    prior_table[6][4][4] = 1./5
    prior_table[6][4][5] = 0.
    prior_table[6][4][6] = 0.
    prior_table[6][4][7] = 1./5
    prior_table[6][4][8] = 1./5
    prior_table[6][4][9] = 0.
    prior_table[6][4][10] = 0.
    prior_table[6][4][11] = 0.
    prior_table[6][4][12] = 0.

    prior_table[7][4][0] = 1./5
    prior_table[7][4][1] = 0.
    prior_table[7][4][2] = 0.
    prior_table[7][4][3] = 1./5
    prior_table[7][4][4] = 1./5
    prior_table[7][4][5] = 0.
    prior_table[7][4][6] = 0.
    prior_table[7][4][7] = 1./5
    prior_table[7][4][8] = 1./5
    prior_table[7][4][9] = 0.
    prior_table[7][4][10] = 0.
    prior_table[7][4][11] = 0.
    prior_table[7][4][12] = 0.

    prior_table[8][4][0] = 1./5
    prior_table[8][4][1] = 0.
    prior_table[8][4][2] = 0.
    prior_table[8][4][3] = 1./5
    prior_table[8][4][4] = 1./5
    prior_table[8][4][5] = 0.
    prior_table[8][4][6] = 0.
    prior_table[8][4][7] = 1./5
    prior_table[8][4][8] = 1./5
    prior_table[8][4][9] = 0.
    prior_table[8][4][10] = 0.
    prior_table[8][4][11] = 0.
    prior_table[8][4][12] = 0.

    prior_table[9][4][0] = 1./5
    prior_table[9][4][1] = 0.
    prior_table[9][4][2] = 0.
    prior_table[9][4][3] = 1./5
    prior_table[9][4][4] = 1./5
    prior_table[9][4][5] = 0.
    prior_table[9][4][6] = 0.
    prior_table[9][4][7] = 1./5
    prior_table[9][4][8] = 1./5
    prior_table[9][4][9] = 0.
    prior_table[9][4][10] = 0.
    prior_table[9][4][11] = 0.
    prior_table[9][4][12] = 0.

    prior_table[10][4][0] = 1./5
    prior_table[10][4][1] = 0.
    prior_table[10][4][2] = 0.
    prior_table[10][4][3] = 1./5
    prior_table[10][4][4] = 1./5
    prior_table[10][4][5] = 0.
    prior_table[10][4][6] = 0.
    prior_table[10][4][7] = 1./5
    prior_table[10][4][8] = 1./5
    prior_table[10][4][9] = 0.
    prior_table[10][4][10] = 0.
    prior_table[10][4][11] = 0.
    prior_table[10][4][12] = 0.

    prior_table[11][4][0] = 1./5
    prior_table[11][4][1] = 0.
    prior_table[11][4][2] = 0.
    prior_table[11][4][3] = 1./5
    prior_table[11][4][4] = 1./5
    prior_table[11][4][5] = 0.
    prior_table[11][4][6] = 0.
    prior_table[11][4][7] = 1./5
    prior_table[11][4][8] = 1./5
    prior_table[11][4][9] = 0.
    prior_table[11][4][10] = 0.
    prior_table[11][4][11] = 0.
    prior_table[11][4][12] = 0.

    prior_table[12][4][0] = 1./5
    prior_table[12][4][1] = 0.
    prior_table[12][4][2] = 0.
    prior_table[12][4][3] = 1./5
    prior_table[12][4][4] = 1./5
    prior_table[12][4][5] = 0.
    prior_table[12][4][6] = 0.
    prior_table[12][4][7] = 1./5
    prior_table[12][4][8] = 1./5
    prior_table[12][4][9] = 0.
    prior_table[12][4][10] = 0.
    prior_table[12][4][11] = 0.
    prior_table[12][4][12] = 0.

    #################################################################

    prior_table[0][5][0] = 1./7
    prior_table[0][5][1] = 1./7
    prior_table[0][5][2] = 1./7
    prior_table[0][5][3] = 0.
    prior_table[0][5][4] = 0.
    prior_table[0][5][5] = 1./7
    prior_table[0][5][6] = 1./7
    prior_table[0][5][7] = 0.
    prior_table[0][5][8] = 0.
    prior_table[0][5][9] = 1./7
    prior_table[0][5][10] = 1./7
    prior_table[0][5][11] = 0.
    prior_table[0][5][12] = 0.

    prior_table[1][5][0] = 1./7
    prior_table[1][5][1] = 1./7
    prior_table[1][5][2] = 1./7
    prior_table[1][5][3] = 0.
    prior_table[1][5][4] = 0.
    prior_table[1][5][5] = 1./7
    prior_table[1][5][6] = 1./7
    prior_table[1][5][7] = 0.
    prior_table[1][5][8] = 0.
    prior_table[1][5][9] = 1./7
    prior_table[1][5][10] = 1./7
    prior_table[1][5][11] = 0.
    prior_table[1][5][12] = 0.

    prior_table[2][5][0] = 1./7
    prior_table[2][5][1] = 1./7
    prior_table[2][5][2] = 1./7
    prior_table[2][5][3] = 0.
    prior_table[2][5][4] = 0.
    prior_table[2][5][5] = 1./7
    prior_table[2][5][6] = 1./7
    prior_table[2][5][7] = 0.
    prior_table[2][5][8] = 0.
    prior_table[2][5][9] = 1./7
    prior_table[2][5][10] = 1./7
    prior_table[2][5][11] = 0.
    prior_table[2][5][12] = 0.

    prior_table[3][5][0] = 1./7
    prior_table[3][5][1] = 1./7
    prior_table[3][5][2] = 1./7
    prior_table[3][5][3] = 0.
    prior_table[3][5][4] = 0.
    prior_table[3][5][5] = 1./7
    prior_table[3][5][6] = 1./7
    prior_table[3][5][7] = 0.
    prior_table[3][5][8] = 0.
    prior_table[3][5][9] = 1./7
    prior_table[3][5][10] = 1./7
    prior_table[3][5][11] = 0.
    prior_table[3][5][12] = 0.

    prior_table[4][5][0] = 1./7
    prior_table[4][5][1] = 1./7
    prior_table[4][5][2] = 1./7
    prior_table[4][5][3] = 0.
    prior_table[4][5][4] = 0.
    prior_table[4][5][5] = 1./7
    prior_table[4][5][6] = 1./7
    prior_table[4][5][7] = 0.
    prior_table[4][5][8] = 0.
    prior_table[4][5][9] = 1./7
    prior_table[4][5][10] = 1./7
    prior_table[4][5][11] = 0.
    prior_table[4][5][12] = 0.

    prior_table[5][5][0] = 1./7
    prior_table[5][5][1] = 1./7
    prior_table[5][5][2] = 1./7
    prior_table[5][5][3] = 0.
    prior_table[5][5][4] = 0.
    prior_table[5][5][5] = 1./7
    prior_table[5][5][6] = 1./7
    prior_table[5][5][7] = 0.
    prior_table[5][5][8] = 0.
    prior_table[5][5][9] = 1./7
    prior_table[5][5][10] = 1./7
    prior_table[5][5][11] = 0.
    prior_table[5][5][12] = 0.

    prior_table[6][5][0] = 1./7
    prior_table[6][5][1] = 1./7
    prior_table[6][5][2] = 1./7
    prior_table[6][5][3] = 0.
    prior_table[6][5][4] = 0.
    prior_table[6][5][5] = 1./7
    prior_table[6][5][6] = 1./7
    prior_table[6][5][7] = 0.
    prior_table[6][5][8] = 0.
    prior_table[6][5][9] = 1./7
    prior_table[6][5][10] = 1./7
    prior_table[6][5][11] = 0.
    prior_table[6][5][12] = 0.

    prior_table[7][5][0] = 1./7
    prior_table[7][5][1] = 1./7
    prior_table[7][5][2] = 1./7
    prior_table[7][5][3] = 0.
    prior_table[7][5][4] = 0.
    prior_table[7][5][5] = 1./7
    prior_table[7][5][6] = 1./7
    prior_table[7][5][7] = 0.
    prior_table[7][5][8] = 0.
    prior_table[7][5][9] = 1./7
    prior_table[7][5][10] = 1./7
    prior_table[7][5][11] = 0.
    prior_table[7][5][12] = 0.

    prior_table[8][5][0] = 1./7
    prior_table[8][5][1] = 1./7
    prior_table[8][5][2] = 1./7
    prior_table[8][5][3] = 0.
    prior_table[8][5][4] = 0.
    prior_table[8][5][5] = 1./7
    prior_table[8][5][6] = 1./7
    prior_table[8][5][7] = 0.
    prior_table[8][5][8] = 0.
    prior_table[8][5][9] = 1./7
    prior_table[8][5][10] = 1./7
    prior_table[8][5][11] = 0.
    prior_table[8][5][12] = 0.

    prior_table[9][5][0] = 1./7
    prior_table[9][5][1] = 1./7
    prior_table[9][5][2] = 1./7
    prior_table[9][5][3] = 0.
    prior_table[9][5][4] = 0.
    prior_table[9][5][5] = 1./7
    prior_table[9][5][6] = 1./7
    prior_table[9][5][7] = 0.
    prior_table[9][5][8] = 0.
    prior_table[9][5][9] = 1./7
    prior_table[9][5][10] = 1./7
    prior_table[9][5][11] = 0.
    prior_table[9][5][12] = 0.

    prior_table[10][5][0] = 1./7
    prior_table[10][5][1] = 1./7
    prior_table[10][5][2] = 1./7
    prior_table[10][5][3] = 0.
    prior_table[10][5][4] = 0.
    prior_table[10][5][5] = 1./7
    prior_table[10][5][6] = 1./7
    prior_table[10][5][7] = 0.
    prior_table[10][5][8] = 0.
    prior_table[10][5][9] = 1./7
    prior_table[10][5][10] = 1./7
    prior_table[10][5][11] = 0.
    prior_table[10][5][12] = 0.

    prior_table[11][5][0] = 1./7
    prior_table[11][5][1] = 1./7
    prior_table[11][5][2] = 1./7
    prior_table[11][5][3] = 0.
    prior_table[11][5][4] = 0.
    prior_table[11][5][5] = 1./7
    prior_table[11][5][6] = 1./7
    prior_table[11][5][7] = 0.
    prior_table[11][5][8] = 0.
    prior_table[11][5][9] = 1./7
    prior_table[11][5][10] = 1./7
    prior_table[11][5][11] = 0.
    prior_table[11][5][12] = 0.

    prior_table[12][5][0] = 1./7
    prior_table[12][5][1] = 1./7
    prior_table[12][5][2] = 1./7
    prior_table[12][5][3] = 0.
    prior_table[12][5][4] = 0.
    prior_table[12][5][5] = 1./7
    prior_table[12][5][6] = 1./7
    prior_table[12][5][7] = 0.
    prior_table[12][5][8] = 0.
    prior_table[12][5][9] = 1./7
    prior_table[12][5][10] = 1./7
    prior_table[12][5][11] = 0.
    prior_table[12][5][12] = 0.

    ################################################################

    prior_table[0][6][0] = 0.
    prior_table[0][6][1] = 1./9
    prior_table[0][6][2] = 1./9
    prior_table[0][6][3] = 1./9
    prior_table[0][6][4] = 0.
    prior_table[0][6][5] = 1./9
    prior_table[0][6][6] = 1./9
    prior_table[0][6][7] = 1./9
    prior_table[0][6][8] = 0.
    prior_table[0][6][9] = 1./9
    prior_table[0][6][10] = 1./9
    prior_table[0][6][11] = 1./9
    prior_table[0][6][12] = 0.

    prior_table[1][6][0] = 0.
    prior_table[1][6][1] = 1./9
    prior_table[1][6][2] = 1./9
    prior_table[1][6][3] = 1./9
    prior_table[1][6][4] = 0.
    prior_table[1][6][5] = 1./9
    prior_table[1][6][6] = 1./9
    prior_table[1][6][7] = 1./9
    prior_table[1][6][8] = 0.
    prior_table[1][6][9] = 1./9
    prior_table[1][6][10] = 1./9
    prior_table[1][6][11] = 1./9
    prior_table[1][6][12] = 0.

    prior_table[2][6][0] = 0.
    prior_table[2][6][1] = 1./9
    prior_table[2][6][2] = 1./9
    prior_table[2][6][3] = 1./9
    prior_table[2][6][4] = 0.
    prior_table[2][6][5] = 1./9
    prior_table[2][6][6] = 1./9
    prior_table[2][6][7] = 1./9
    prior_table[2][6][8] = 0.
    prior_table[2][6][9] = 1./9
    prior_table[2][6][10] = 1./9
    prior_table[2][6][11] = 1./9
    prior_table[2][6][12] = 0.

    prior_table[3][6][0] = 0.
    prior_table[3][6][1] = 1./9
    prior_table[3][6][2] = 1./9
    prior_table[3][6][3] = 1./9
    prior_table[3][6][4] = 0.
    prior_table[3][6][5] = 1./9
    prior_table[3][6][6] = 1./9
    prior_table[3][6][7] = 1./9
    prior_table[3][6][8] = 0.
    prior_table[3][6][9] = 1./9
    prior_table[3][6][10] = 1./9
    prior_table[3][6][11] = 1./9
    prior_table[3][6][12] = 0.

    prior_table[4][6][0] = 0.
    prior_table[4][6][1] = 1./9
    prior_table[4][6][2] = 1./9
    prior_table[4][6][3] = 1./9
    prior_table[4][6][4] = 0.
    prior_table[4][6][5] = 1./9
    prior_table[4][6][6] = 1./9
    prior_table[4][6][7] = 1./9
    prior_table[4][6][8] = 0.
    prior_table[4][6][9] = 1./9
    prior_table[4][6][10] = 1./9
    prior_table[4][6][11] = 1./9
    prior_table[4][6][12] = 0.

    prior_table[5][6][0] = 0.
    prior_table[5][6][1] = 1./9
    prior_table[5][6][2] = 1./9
    prior_table[5][6][3] = 1./9
    prior_table[5][6][4] = 0.
    prior_table[5][6][5] = 1./9
    prior_table[5][6][6] = 1./9
    prior_table[5][6][7] = 1./9
    prior_table[5][6][8] = 0.
    prior_table[5][6][9] = 1./9
    prior_table[5][6][10] = 1./9
    prior_table[5][6][11] = 1./9
    prior_table[5][6][12] = 0.

    prior_table[6][6][0] = 0.
    prior_table[6][6][1] = 1./9
    prior_table[6][6][2] = 1./9
    prior_table[6][6][3] = 1./9
    prior_table[6][6][4] = 0.
    prior_table[6][6][5] = 1./9
    prior_table[6][6][6] = 1./9
    prior_table[6][6][7] = 1./9
    prior_table[6][6][8] = 0.
    prior_table[6][6][9] = 1./9
    prior_table[6][6][10] = 1./9
    prior_table[6][6][11] = 1./9
    prior_table[6][6][12] = 0.

    prior_table[7][6][0] = 0.
    prior_table[7][6][1] = 1./9
    prior_table[7][6][2] = 1./9
    prior_table[7][6][3] = 1./9
    prior_table[7][6][4] = 0.
    prior_table[7][6][5] = 1./9
    prior_table[7][6][6] = 1./9
    prior_table[7][6][7] = 1./9
    prior_table[7][6][8] = 0.
    prior_table[7][6][9] = 1./9
    prior_table[7][6][10] = 1./9
    prior_table[7][6][11] = 1./9
    prior_table[7][6][12] = 0.

    prior_table[8][6][0] = 0.
    prior_table[8][6][1] = 1./9
    prior_table[8][6][2] = 1./9
    prior_table[8][6][3] = 1./9
    prior_table[8][6][4] = 0.
    prior_table[8][6][5] = 1./9
    prior_table[8][6][6] = 1./9
    prior_table[8][6][7] = 1./9
    prior_table[8][6][8] = 0.
    prior_table[8][6][9] = 1./9
    prior_table[8][6][10] = 1./9
    prior_table[8][6][11] = 1./9
    prior_table[8][6][12] = 0.

    prior_table[9][6][0] = 0.
    prior_table[9][6][1] = 1./9
    prior_table[9][6][2] = 1./9
    prior_table[9][6][3] = 1./9
    prior_table[9][6][4] = 0.
    prior_table[9][6][5] = 1./9
    prior_table[9][6][6] = 1./9
    prior_table[9][6][7] = 1./9
    prior_table[9][6][8] = 0.
    prior_table[9][6][9] = 1./9
    prior_table[9][6][10] = 1./9
    prior_table[9][6][11] = 1./9
    prior_table[9][6][12] = 0.

    prior_table[10][6][0] = 0.
    prior_table[10][6][1] = 1./9
    prior_table[10][6][2] = 1./9
    prior_table[10][6][3] = 1./9
    prior_table[10][6][4] = 0.
    prior_table[10][6][5] = 1./9
    prior_table[10][6][6] = 1./9
    prior_table[10][6][7] = 1./9
    prior_table[10][6][8] = 0.
    prior_table[10][6][9] = 1./9
    prior_table[10][6][10] = 1./9
    prior_table[10][6][11] = 1./9
    prior_table[10][6][12] = 0.

    prior_table[11][6][0] = 0.
    prior_table[11][6][1] = 1./9
    prior_table[11][6][2] = 1./9
    prior_table[11][6][3] = 1./9
    prior_table[11][6][4] = 0.
    prior_table[11][6][5] = 1./9
    prior_table[11][6][6] = 1./9
    prior_table[11][6][7] = 1./9
    prior_table[11][6][8] = 0.
    prior_table[11][6][9] = 1./9
    prior_table[11][6][10] = 1./9
    prior_table[11][6][11] = 1./9
    prior_table[11][6][12] = 0.

    prior_table[12][6][0] = 0.
    prior_table[12][6][1] = 1./9
    prior_table[12][6][2] = 1./9
    prior_table[12][6][3] = 1./9
    prior_table[12][6][4] = 0.
    prior_table[12][6][5] = 1./9
    prior_table[12][6][6] = 1./9
    prior_table[12][6][7] = 1./9
    prior_table[12][6][8] = 0.
    prior_table[12][6][9] = 1./9
    prior_table[12][6][10] = 1./9
    prior_table[12][6][11] = 1./9
    prior_table[12][6][12] = 0.

    ################################################################

    prior_table[0][7][0] = 0.
    prior_table[0][7][1] = 0.
    prior_table[0][7][2] = 1./9
    prior_table[0][7][3] = 1./9
    prior_table[0][7][4] = 1./9
    prior_table[0][7][5] = 0.
    prior_table[0][7][6] = 1./9
    prior_table[0][7][7] = 1./9
    prior_table[0][7][8] = 1./9
    prior_table[0][7][9] = 0.
    prior_table[0][7][10] = 1./9
    prior_table[0][7][11] = 1./9
    prior_table[0][7][12] = 1./9

    prior_table[1][7][0] = 0.
    prior_table[1][7][1] = 0.
    prior_table[1][7][2] = 1./9
    prior_table[1][7][3] = 1./9
    prior_table[1][7][4] = 1./9
    prior_table[1][7][5] = 0.
    prior_table[1][7][6] = 1./9
    prior_table[1][7][7] = 1./9
    prior_table[1][7][8] = 1./9
    prior_table[1][7][9] = 0.
    prior_table[1][7][10] = 1./9
    prior_table[1][7][11] = 1./9
    prior_table[1][7][12] = 1./9

    prior_table[2][7][0] = 0.
    prior_table[2][7][1] = 0.
    prior_table[2][7][2] = 1./9
    prior_table[2][7][3] = 1./9
    prior_table[2][7][4] = 1./9
    prior_table[2][7][5] = 0.
    prior_table[2][7][6] = 1./9
    prior_table[2][7][7] = 1./9
    prior_table[2][7][8] = 1./9
    prior_table[2][7][9] = 0.
    prior_table[2][7][10] = 1./9
    prior_table[2][7][11] = 1./9
    prior_table[2][7][12] = 1./9

    prior_table[3][7][0] = 0.
    prior_table[3][7][1] = 0.
    prior_table[3][7][2] = 1./9
    prior_table[3][7][3] = 1./9
    prior_table[3][7][4] = 1./9
    prior_table[3][7][5] = 0.
    prior_table[3][7][6] = 1./9
    prior_table[3][7][7] = 1./9
    prior_table[3][7][8] = 1./9
    prior_table[3][7][9] = 0.
    prior_table[3][7][10] = 1./9
    prior_table[3][7][11] = 1./9
    prior_table[3][7][12] = 1./9

    prior_table[4][7][0] = 0.
    prior_table[4][7][1] = 0.
    prior_table[4][7][2] = 1./9
    prior_table[4][7][3] = 1./9
    prior_table[4][7][4] = 1./9
    prior_table[4][7][5] = 0.
    prior_table[4][7][6] = 1./9
    prior_table[4][7][7] = 1./9
    prior_table[4][7][8] = 1./9
    prior_table[4][7][9] = 0.
    prior_table[4][7][10] = 1./9
    prior_table[4][7][11] = 1./9
    prior_table[4][7][12] = 1./9

    prior_table[5][7][0] = 0.
    prior_table[5][7][1] = 0.
    prior_table[5][7][2] = 1./9
    prior_table[5][7][3] = 1./9
    prior_table[5][7][4] = 1./9
    prior_table[5][7][5] = 0.
    prior_table[5][7][6] = 1./9
    prior_table[5][7][7] = 1./9
    prior_table[5][7][8] = 1./9
    prior_table[5][7][9] = 0.
    prior_table[5][7][10] = 1./9
    prior_table[5][7][11] = 1./9
    prior_table[5][7][12] = 1./9

    prior_table[6][7][0] = 0.
    prior_table[6][7][1] = 0.
    prior_table[6][7][2] = 1./9
    prior_table[6][7][3] = 1./9
    prior_table[6][7][4] = 1./9
    prior_table[6][7][5] = 0.
    prior_table[6][7][6] = 1./9
    prior_table[6][7][7] = 1./9
    prior_table[6][7][8] = 1./9
    prior_table[6][7][9] = 0.
    prior_table[6][7][10] = 1./9
    prior_table[6][7][11] = 1./9
    prior_table[6][7][12] = 1./9

    prior_table[7][7][0] = 0.
    prior_table[7][7][1] = 0.
    prior_table[7][7][2] = 1./9
    prior_table[7][7][3] = 1./9
    prior_table[7][7][4] = 1./9
    prior_table[7][7][5] = 0.
    prior_table[7][7][6] = 1./9
    prior_table[7][7][7] = 1./9
    prior_table[7][7][8] = 1./9
    prior_table[7][7][9] = 0.
    prior_table[7][7][10] = 1./9
    prior_table[7][7][11] = 1./9
    prior_table[7][7][12] = 1./9

    prior_table[8][7][0] = 0.
    prior_table[8][7][1] = 0.
    prior_table[8][7][2] = 1./9
    prior_table[8][7][3] = 1./9
    prior_table[8][7][4] = 1./9
    prior_table[8][7][5] = 0.
    prior_table[8][7][6] = 1./9
    prior_table[8][7][7] = 1./9
    prior_table[8][7][8] = 1./9
    prior_table[8][7][9] = 0.
    prior_table[8][7][10] = 1./9
    prior_table[8][7][11] = 1./9
    prior_table[8][7][12] = 1./9

    prior_table[9][7][0] = 0.
    prior_table[9][7][1] = 0.
    prior_table[9][7][2] = 1./9
    prior_table[9][7][3] = 1./9
    prior_table[9][7][4] = 1./9
    prior_table[9][7][5] = 0.
    prior_table[9][7][6] = 1./9
    prior_table[9][7][7] = 1./9
    prior_table[9][7][8] = 1./9
    prior_table[9][7][9] = 0.
    prior_table[9][7][10] = 1./9
    prior_table[9][7][11] = 1./9
    prior_table[9][7][12] = 1./9

    prior_table[10][7][0] = 0.
    prior_table[10][7][1] = 0.
    prior_table[10][7][2] = 1./9
    prior_table[10][7][3] = 1./9
    prior_table[10][7][4] = 1./9
    prior_table[10][7][5] = 0.
    prior_table[10][7][6] = 1./9
    prior_table[10][7][7] = 1./9
    prior_table[10][7][8] = 1./9
    prior_table[10][7][9] = 0.
    prior_table[10][7][10] = 1./9
    prior_table[10][7][11] = 1./9
    prior_table[10][7][12] = 1./9

    prior_table[11][7][0] = 0.
    prior_table[11][7][1] = 0.
    prior_table[11][7][2] = 1./9
    prior_table[11][7][3] = 1./9
    prior_table[11][7][4] = 1./9
    prior_table[11][7][5] = 0.
    prior_table[11][7][6] = 1./9
    prior_table[11][7][7] = 1./9
    prior_table[11][7][8] = 1./9
    prior_table[11][7][9] = 0.
    prior_table[11][7][10] = 1./9
    prior_table[11][7][11] = 1./9
    prior_table[11][7][12] = 1./9

    prior_table[12][7][0] = 0.
    prior_table[12][7][1] = 0.
    prior_table[12][7][2] = 1./9
    prior_table[12][7][3] = 1./9
    prior_table[12][7][4] = 1./9
    prior_table[12][7][5] = 0.
    prior_table[12][7][6] = 1./9
    prior_table[12][7][7] = 1./9
    prior_table[12][7][8] = 1./9
    prior_table[12][7][9] = 0.
    prior_table[12][7][10] = 1./9
    prior_table[12][7][11] = 1./9
    prior_table[12][7][12] = 1./9

    ###############################################################

    prior_table[0][8][0] = 1./7
    prior_table[0][8][1] = 0.
    prior_table[0][8][2] = 0.
    prior_table[0][8][3] = 1./7
    prior_table[0][8][4] = 1./7
    prior_table[0][8][5] = 0.
    prior_table[0][8][6] = 0.
    prior_table[0][8][7] = 1./7
    prior_table[0][8][8] = 1./7
    prior_table[0][8][9] = 0.
    prior_table[0][8][10] = 0.
    prior_table[0][8][11] = 1./7
    prior_table[0][8][12] = 1./7

    prior_table[1][8][0] = 1./7
    prior_table[1][8][1] = 0.
    prior_table[1][8][2] = 0.
    prior_table[1][8][3] = 1./7
    prior_table[1][8][4] = 1./7
    prior_table[1][8][5] = 0.
    prior_table[1][8][6] = 0.
    prior_table[1][8][7] = 1./7
    prior_table[1][8][8] = 1./7
    prior_table[1][8][9] = 0.
    prior_table[1][8][10] = 0.
    prior_table[1][8][11] = 1./7
    prior_table[1][8][12] = 1./7

    prior_table[2][8][0] = 1./7
    prior_table[2][8][1] = 0.
    prior_table[2][8][2] = 0.
    prior_table[2][8][3] = 1./7
    prior_table[2][8][4] = 1./7
    prior_table[2][8][5] = 0.
    prior_table[2][8][6] = 0.
    prior_table[2][8][7] = 1./7
    prior_table[2][8][8] = 1./7
    prior_table[2][8][9] = 0.
    prior_table[2][8][10] = 0.
    prior_table[2][8][11] = 1./7
    prior_table[2][8][12] = 1./7

    prior_table[3][8][0] = 1./7
    prior_table[3][8][1] = 0.
    prior_table[3][8][2] = 0.
    prior_table[3][8][3] = 1./7
    prior_table[3][8][4] = 1./7
    prior_table[3][8][5] = 0.
    prior_table[3][8][6] = 0.
    prior_table[3][8][7] = 1./7
    prior_table[3][8][8] = 1./7
    prior_table[3][8][9] = 0.
    prior_table[3][8][10] = 0.
    prior_table[3][8][11] = 1./7
    prior_table[3][8][12] = 1./7

    prior_table[4][8][0] = 1./7
    prior_table[4][8][1] = 0.
    prior_table[4][8][2] = 0.
    prior_table[4][8][3] = 1./7
    prior_table[4][8][4] = 1./7
    prior_table[4][8][5] = 0.
    prior_table[4][8][6] = 0.
    prior_table[4][8][7] = 1./7
    prior_table[4][8][8] = 1./7
    prior_table[4][8][9] = 0.
    prior_table[4][8][10] = 0.
    prior_table[4][8][11] = 1./7
    prior_table[4][8][12] = 1./7

    prior_table[5][8][0] = 1./7
    prior_table[5][8][1] = 0.
    prior_table[5][8][2] = 0.
    prior_table[5][8][3] = 1./7
    prior_table[5][8][4] = 1./7
    prior_table[5][8][5] = 0.
    prior_table[5][8][6] = 0.
    prior_table[5][8][7] = 1./7
    prior_table[5][8][8] = 1./7
    prior_table[5][8][9] = 0.
    prior_table[5][8][10] = 0.
    prior_table[5][8][11] = 1./7
    prior_table[5][8][12] = 1./7

    prior_table[6][8][0] = 1./7
    prior_table[6][8][1] = 0.
    prior_table[6][8][2] = 0.
    prior_table[6][8][3] = 1./7
    prior_table[6][8][4] = 1./7
    prior_table[6][8][5] = 0.
    prior_table[6][8][6] = 0.
    prior_table[6][8][7] = 1./7
    prior_table[6][8][8] = 1./7
    prior_table[6][8][9] = 0.
    prior_table[6][8][10] = 0.
    prior_table[6][8][11] = 1./7
    prior_table[6][8][12] = 1./7

    prior_table[7][8][0] = 1./7
    prior_table[7][8][1] = 0.
    prior_table[7][8][2] = 0.
    prior_table[7][8][3] = 1./7
    prior_table[7][8][4] = 1./7
    prior_table[7][8][5] = 0.
    prior_table[7][8][6] = 0.
    prior_table[7][8][7] = 1./7
    prior_table[7][8][8] = 1./7
    prior_table[7][8][9] = 0.
    prior_table[7][8][10] = 0.
    prior_table[7][8][11] = 1./7
    prior_table[7][8][12] = 1./7

    prior_table[8][8][0] = 1./7
    prior_table[8][8][1] = 0.
    prior_table[8][8][2] = 0.
    prior_table[8][8][3] = 1./7
    prior_table[8][8][4] = 1./7
    prior_table[8][8][5] = 0.
    prior_table[8][8][6] = 0.
    prior_table[8][8][7] = 1./7
    prior_table[8][8][8] = 1./7
    prior_table[8][8][9] = 0.
    prior_table[8][8][10] = 0.
    prior_table[8][8][11] = 1./7
    prior_table[8][8][12] = 1./7

    prior_table[9][8][0] = 1./7
    prior_table[9][8][1] = 0.
    prior_table[9][8][2] = 0.
    prior_table[9][8][3] = 1./7
    prior_table[9][8][4] = 1./7
    prior_table[9][8][5] = 0.
    prior_table[9][8][6] = 0.
    prior_table[9][8][7] = 1./7
    prior_table[9][8][8] = 1./7
    prior_table[9][8][9] = 0.
    prior_table[9][8][10] = 0.
    prior_table[9][8][11] = 1./7
    prior_table[9][8][12] = 1./7

    prior_table[10][8][0] = 1./7
    prior_table[10][8][1] = 0.
    prior_table[10][8][2] = 0.
    prior_table[10][8][3] = 1./7
    prior_table[10][8][4] = 1./7
    prior_table[10][8][5] = 0.
    prior_table[10][8][6] = 0.
    prior_table[10][8][7] = 1./7
    prior_table[10][8][8] = 1./7
    prior_table[10][8][9] = 0.
    prior_table[10][8][10] = 0.
    prior_table[10][8][11] = 1./7
    prior_table[10][8][12] = 1./7

    prior_table[11][8][0] = 1./7
    prior_table[11][8][1] = 0.
    prior_table[11][8][2] = 0.
    prior_table[11][8][3] = 1./7
    prior_table[11][8][4] = 1./7
    prior_table[11][8][5] = 0.
    prior_table[11][8][6] = 0.
    prior_table[11][8][7] = 1./7
    prior_table[11][8][8] = 1./7
    prior_table[11][8][9] = 0.
    prior_table[11][8][10] = 0.
    prior_table[11][8][11] = 1./7
    prior_table[11][8][12] = 1./7

    prior_table[12][8][0] = 1./7
    prior_table[12][8][1] = 0.
    prior_table[12][8][2] = 0.
    prior_table[12][8][3] = 1./7
    prior_table[12][8][4] = 1./7
    prior_table[12][8][5] = 0.
    prior_table[12][8][6] = 0.
    prior_table[12][8][7] = 1./7
    prior_table[12][8][8] = 1./7
    prior_table[12][8][9] = 0.
    prior_table[12][8][10] = 0.
    prior_table[12][8][11] = 1./7
    prior_table[12][8][12] = 1./7

    ##############################################################

    prior_table[0][9][0] = 1./5
    prior_table[0][9][1] = 0.
    prior_table[0][9][2] = 0.
    prior_table[0][9][3] = 0.
    prior_table[0][9][4] = 0.
    prior_table[0][9][5] = 1./5
    prior_table[0][9][6] = 1./5
    prior_table[0][9][7] = 0.
    prior_table[0][9][8] = 0.
    prior_table[0][9][9] = 1./5
    prior_table[0][9][10] = 1./5
    prior_table[0][9][11] = 0.
    prior_table[0][9][12] = 0.

    prior_table[1][9][0] = 1./5
    prior_table[1][9][1] = 0.
    prior_table[1][9][2] = 0.
    prior_table[1][9][3] = 0.
    prior_table[1][9][4] = 0.
    prior_table[1][9][5] = 1./5
    prior_table[1][9][6] = 1./5
    prior_table[1][9][7] = 0.
    prior_table[1][9][8] = 0.
    prior_table[1][9][9] = 1./5
    prior_table[1][9][10] = 1./5
    prior_table[1][9][11] = 0.
    prior_table[1][9][12] = 0.

    prior_table[2][9][0] = 1./5
    prior_table[2][9][1] = 0.
    prior_table[2][9][2] = 0.
    prior_table[2][9][3] = 0.
    prior_table[2][9][4] = 0.
    prior_table[2][9][5] = 1./5
    prior_table[2][9][6] = 1./5
    prior_table[2][9][7] = 0.
    prior_table[2][9][8] = 0.
    prior_table[2][9][9] = 1./5
    prior_table[2][9][10] = 1./5
    prior_table[2][9][11] = 0.
    prior_table[2][9][12] = 0.

    prior_table[3][9][0] = 1./5
    prior_table[3][9][1] = 0.
    prior_table[3][9][2] = 0.
    prior_table[3][9][3] = 0.
    prior_table[3][9][4] = 0.
    prior_table[3][9][5] = 1./5
    prior_table[3][9][6] = 1./5
    prior_table[3][9][7] = 0.
    prior_table[3][9][8] = 0.
    prior_table[3][9][9] = 1./5
    prior_table[3][9][10] = 1./5
    prior_table[3][9][11] = 0.
    prior_table[3][9][12] = 0.

    prior_table[4][9][0] = 1./5
    prior_table[4][9][1] = 0.
    prior_table[4][9][2] = 0.
    prior_table[4][9][3] = 0.
    prior_table[4][9][4] = 0.
    prior_table[4][9][5] = 1./5
    prior_table[4][9][6] = 1./5
    prior_table[4][9][7] = 0.
    prior_table[4][9][8] = 0.
    prior_table[4][9][9] = 1./5
    prior_table[4][9][10] = 1./5
    prior_table[4][9][11] = 0.
    prior_table[4][9][12] = 0.

    prior_table[5][9][0] = 1./5
    prior_table[5][9][1] = 0.
    prior_table[5][9][2] = 0.
    prior_table[5][9][3] = 0.
    prior_table[5][9][4] = 0.
    prior_table[5][9][5] = 1./5
    prior_table[5][9][6] = 1./5
    prior_table[5][9][7] = 0.
    prior_table[5][9][8] = 0.
    prior_table[5][9][9] = 1./5
    prior_table[5][9][10] = 1./5
    prior_table[5][9][11] = 0.
    prior_table[5][9][12] = 0.

    prior_table[6][9][0] = 1./5
    prior_table[6][9][1] = 0.
    prior_table[6][9][2] = 0.
    prior_table[6][9][3] = 0.
    prior_table[6][9][4] = 0.
    prior_table[6][9][5] = 1./5
    prior_table[6][9][6] = 1./5
    prior_table[6][9][7] = 0.
    prior_table[6][9][8] = 0.
    prior_table[6][9][9] = 1./5
    prior_table[6][9][10] = 1./5
    prior_table[6][9][11] = 0.
    prior_table[6][9][12] = 0.

    prior_table[7][9][0] = 1./5
    prior_table[7][9][1] = 0.
    prior_table[7][9][2] = 0.
    prior_table[7][9][3] = 0.
    prior_table[7][9][4] = 0.
    prior_table[7][9][5] = 1./5
    prior_table[7][9][6] = 1./5
    prior_table[7][9][7] = 0.
    prior_table[7][9][8] = 0.
    prior_table[7][9][9] = 1./5
    prior_table[7][9][10] = 1./5
    prior_table[7][9][11] = 0.
    prior_table[7][9][12] = 0.

    prior_table[8][9][0] = 1./5
    prior_table[8][9][1] = 0.
    prior_table[8][9][2] = 0.
    prior_table[8][9][3] = 0.
    prior_table[8][9][4] = 0.
    prior_table[8][9][5] = 1./5
    prior_table[8][9][6] = 1./5
    prior_table[8][9][7] = 0.
    prior_table[8][9][8] = 0.
    prior_table[8][9][9] = 1./5
    prior_table[8][9][10] = 1./5
    prior_table[8][9][11] = 0.
    prior_table[8][9][12] = 0.

    prior_table[9][9][0] = 1./5
    prior_table[9][9][1] = 0.
    prior_table[9][9][2] = 0.
    prior_table[9][9][3] = 0.
    prior_table[9][9][4] = 0.
    prior_table[9][9][5] = 1./5
    prior_table[9][9][6] = 1./5
    prior_table[9][9][7] = 0.
    prior_table[9][9][8] = 0.
    prior_table[9][9][9] = 1./5
    prior_table[9][9][10] = 1./5
    prior_table[9][9][11] = 0.
    prior_table[9][9][12] = 0.

    prior_table[10][9][0] = 1./5
    prior_table[10][9][1] = 0.
    prior_table[10][9][2] = 0.
    prior_table[10][9][3] = 0.
    prior_table[10][9][4] = 0.
    prior_table[10][9][5] = 1./5
    prior_table[10][9][6] = 1./5
    prior_table[10][9][7] = 0.
    prior_table[10][9][8] = 0.
    prior_table[10][9][9] = 1./5
    prior_table[10][9][10] = 1./5
    prior_table[10][9][11] = 0.
    prior_table[10][9][12] = 0.

    prior_table[11][9][0] = 1./5
    prior_table[11][9][1] = 0.
    prior_table[11][9][2] = 0.
    prior_table[11][9][3] = 0.
    prior_table[11][9][4] = 0.
    prior_table[11][9][5] = 1./5
    prior_table[11][9][6] = 1./5
    prior_table[11][9][7] = 0.
    prior_table[11][9][8] = 0.
    prior_table[11][9][9] = 1./5
    prior_table[11][9][10] = 1./5
    prior_table[11][9][11] = 0.
    prior_table[11][9][12] = 0.

    prior_table[12][9][0] = 1./5
    prior_table[12][9][1] = 0.
    prior_table[12][9][2] = 0.
    prior_table[12][9][3] = 0.
    prior_table[12][9][4] = 0.
    prior_table[12][9][5] = 1./5
    prior_table[12][9][6] = 1./5
    prior_table[12][9][7] = 0.
    prior_table[12][9][8] = 0.
    prior_table[12][9][9] = 1./5
    prior_table[12][9][10] = 1./5
    prior_table[12][9][11] = 0.
    prior_table[12][9][12] = 0.

    ###########################################################

    prior_table[0][10][0] = 1./7
    prior_table[0][10][1] = 0.
    prior_table[0][10][2] = 0.
    prior_table[0][10][3] = 0.
    prior_table[0][10][4] = 0.
    prior_table[0][10][5] = 1./7
    prior_table[0][10][6] = 1./7
    prior_table[0][10][7] = 1./7
    prior_table[0][10][8] = 0.
    prior_table[0][10][9] = 1./7
    prior_table[0][10][10] = 1./7
    prior_table[0][10][11] = 1./7
    prior_table[0][10][12] = 0.

    prior_table[1][10][0] = 1./7
    prior_table[1][10][1] = 0.
    prior_table[1][10][2] = 0.
    prior_table[1][10][3] = 0.
    prior_table[1][10][4] = 0.
    prior_table[1][10][5] = 1./7
    prior_table[1][10][6] = 1./7
    prior_table[1][10][7] = 1./7
    prior_table[1][10][8] = 0.
    prior_table[1][10][9] = 1./7
    prior_table[1][10][10] = 1./7
    prior_table[1][10][11] = 1./7
    prior_table[1][10][12] = 0.

    prior_table[2][10][0] = 1./7
    prior_table[2][10][1] = 0.
    prior_table[2][10][2] = 0.
    prior_table[2][10][3] = 0.
    prior_table[2][10][4] = 0.
    prior_table[2][10][5] = 1./7
    prior_table[2][10][6] = 1./7
    prior_table[2][10][7] = 1./7
    prior_table[2][10][8] = 0.
    prior_table[2][10][9] = 1./7
    prior_table[2][10][10] = 1./7
    prior_table[2][10][11] = 1./7
    prior_table[2][10][12] = 0.

    prior_table[3][10][0] = 1./7
    prior_table[3][10][1] = 0.
    prior_table[3][10][2] = 0.
    prior_table[3][10][3] = 0.
    prior_table[3][10][4] = 0.
    prior_table[3][10][5] = 1./7
    prior_table[3][10][6] = 1./7
    prior_table[3][10][7] = 1./7
    prior_table[3][10][8] = 0.
    prior_table[3][10][9] = 1./7
    prior_table[3][10][10] = 1./7
    prior_table[3][10][11] = 1./7
    prior_table[3][10][12] = 0.

    prior_table[4][10][0] = 1./7
    prior_table[4][10][1] = 0.
    prior_table[4][10][2] = 0.
    prior_table[4][10][3] = 0.
    prior_table[4][10][4] = 0.
    prior_table[4][10][5] = 1./7
    prior_table[4][10][6] = 1./7
    prior_table[4][10][7] = 1./7
    prior_table[4][10][8] = 0.
    prior_table[4][10][9] = 1./7
    prior_table[4][10][10] = 1./7
    prior_table[4][10][11] = 1./7
    prior_table[4][10][12] = 0.

    prior_table[5][10][0] = 1./7
    prior_table[5][10][1] = 0.
    prior_table[5][10][2] = 0.
    prior_table[5][10][3] = 0.
    prior_table[5][10][4] = 0.
    prior_table[5][10][5] = 1./7
    prior_table[5][10][6] = 1./7
    prior_table[5][10][7] = 1./7
    prior_table[5][10][8] = 0.
    prior_table[5][10][9] = 1./7
    prior_table[5][10][10] = 1./7
    prior_table[5][10][11] = 1./7
    prior_table[5][10][12] = 0.

    prior_table[6][10][0] = 1./7
    prior_table[6][10][1] = 0.
    prior_table[6][10][2] = 0.
    prior_table[6][10][3] = 0.
    prior_table[6][10][4] = 0.
    prior_table[6][10][5] = 1./7
    prior_table[6][10][6] = 1./7
    prior_table[6][10][7] = 1./7
    prior_table[6][10][8] = 0.
    prior_table[6][10][9] = 1./7
    prior_table[6][10][10] = 1./7
    prior_table[6][10][11] = 1./7
    prior_table[6][10][12] = 0.

    prior_table[7][10][0] = 1./7
    prior_table[7][10][1] = 0.
    prior_table[7][10][2] = 0.
    prior_table[7][10][3] = 0.
    prior_table[7][10][4] = 0.
    prior_table[7][10][5] = 1./7
    prior_table[7][10][6] = 1./7
    prior_table[7][10][7] = 1./7
    prior_table[7][10][8] = 0.
    prior_table[7][10][9] = 1./7
    prior_table[7][10][10] = 1./7
    prior_table[7][10][11] = 1./7
    prior_table[7][10][12] = 0.

    prior_table[8][10][0] = 1./7
    prior_table[8][10][1] = 0.
    prior_table[8][10][2] = 0.
    prior_table[8][10][3] = 0.
    prior_table[8][10][4] = 0.
    prior_table[8][10][5] = 1./7
    prior_table[8][10][6] = 1./7
    prior_table[8][10][7] = 1./7
    prior_table[8][10][8] = 0.
    prior_table[8][10][9] = 1./7
    prior_table[8][10][10] = 1./7
    prior_table[8][10][11] = 1./7
    prior_table[8][10][12] = 0.

    prior_table[9][10][0] = 1./7
    prior_table[9][10][1] = 0.
    prior_table[9][10][2] = 0.
    prior_table[9][10][3] = 0.
    prior_table[9][10][4] = 0.
    prior_table[9][10][5] = 1./7
    prior_table[9][10][6] = 1./7
    prior_table[9][10][7] = 1./7
    prior_table[9][10][8] = 0.
    prior_table[9][10][9] = 1./7
    prior_table[9][10][10] = 1./7
    prior_table[9][10][11] = 1./7
    prior_table[9][10][12] = 0.

    prior_table[10][10][0] = 1./7
    prior_table[10][10][1] = 0.
    prior_table[10][10][2] = 0.
    prior_table[10][10][3] = 0.
    prior_table[10][10][4] = 0.
    prior_table[10][10][5] = 1./7
    prior_table[10][10][6] = 1./7
    prior_table[10][10][7] = 1./7
    prior_table[10][10][8] = 0.
    prior_table[10][10][9] = 1./7
    prior_table[10][10][10] = 1./7
    prior_table[10][10][11] = 1./7
    prior_table[10][10][12] = 0.

    prior_table[11][10][0] = 1./7
    prior_table[11][10][1] = 0.
    prior_table[11][10][2] = 0.
    prior_table[11][10][3] = 0.
    prior_table[11][10][4] = 0.
    prior_table[11][10][5] = 1./7
    prior_table[11][10][6] = 1./7
    prior_table[11][10][7] = 1./7
    prior_table[11][10][8] = 0.
    prior_table[11][10][9] = 1./7
    prior_table[11][10][10] = 1./7
    prior_table[11][10][11] = 1./7
    prior_table[11][10][12] = 0.

    prior_table[12][10][0] = 1./7
    prior_table[12][10][1] = 0.
    prior_table[12][10][2] = 0.
    prior_table[12][10][3] = 0.
    prior_table[12][10][4] = 0.
    prior_table[12][10][5] = 1./7
    prior_table[12][10][6] = 1./7
    prior_table[12][10][7] = 1./7
    prior_table[12][10][8] = 0.
    prior_table[12][10][9] = 1./7
    prior_table[12][10][10] = 1./7
    prior_table[12][10][11] = 1./7
    prior_table[12][10][12] = 0.

    #########################################################

    prior_table[0][11][0] = 1./7
    prior_table[0][11][1] = 0.
    prior_table[0][11][2] = 0.
    prior_table[0][11][3] = 0.
    prior_table[0][11][4] = 0.
    prior_table[0][11][5] = 0.
    prior_table[0][11][6] = 1./7
    prior_table[0][11][7] = 1./7
    prior_table[0][11][8] = 1./7
    prior_table[0][11][9] = 0.
    prior_table[0][11][10] = 1./7
    prior_table[0][11][11] = 1./7
    prior_table[0][11][12] = 1./7

    prior_table[1][11][0] = 1./7
    prior_table[1][11][1] = 0.
    prior_table[1][11][2] = 0.
    prior_table[1][11][3] = 0.
    prior_table[1][11][4] = 0.
    prior_table[1][11][5] = 0.
    prior_table[1][11][6] = 1./7
    prior_table[1][11][7] = 1./7
    prior_table[1][11][8] = 1./7
    prior_table[1][11][9] = 0.
    prior_table[1][11][10] = 1./7
    prior_table[1][11][11] = 1./7
    prior_table[1][11][12] = 1./7

    prior_table[2][11][0] = 1./7
    prior_table[2][11][1] = 0.
    prior_table[2][11][2] = 0.
    prior_table[2][11][3] = 0.
    prior_table[2][11][4] = 0.
    prior_table[2][11][5] = 0.
    prior_table[2][11][6] = 1./7
    prior_table[2][11][7] = 1./7
    prior_table[2][11][8] = 1./7
    prior_table[2][11][9] = 0.
    prior_table[2][11][10] = 1./7
    prior_table[2][11][11] = 1./7
    prior_table[2][11][12] = 1./7

    prior_table[3][11][0] = 1./7
    prior_table[3][11][1] = 0.
    prior_table[3][11][2] = 0.
    prior_table[3][11][3] = 0.
    prior_table[3][11][4] = 0.
    prior_table[3][11][5] = 0.
    prior_table[3][11][6] = 1./7
    prior_table[3][11][7] = 1./7
    prior_table[3][11][8] = 1./7
    prior_table[3][11][9] = 0.
    prior_table[3][11][10] = 1./7
    prior_table[3][11][11] = 1./7
    prior_table[3][11][12] = 1./7

    prior_table[4][11][0] = 1./7
    prior_table[4][11][1] = 0.
    prior_table[4][11][2] = 0.
    prior_table[4][11][3] = 0.
    prior_table[4][11][4] = 0.
    prior_table[4][11][5] = 0.
    prior_table[4][11][6] = 1./7
    prior_table[4][11][7] = 1./7
    prior_table[4][11][8] = 1./7
    prior_table[4][11][9] = 0.
    prior_table[4][11][10] = 1./7
    prior_table[4][11][11] = 1./7
    prior_table[4][11][12] = 1./7

    prior_table[5][11][0] = 1./7
    prior_table[5][11][1] = 0.
    prior_table[5][11][2] = 0.
    prior_table[5][11][3] = 0.
    prior_table[5][11][4] = 0.
    prior_table[5][11][5] = 0.
    prior_table[5][11][6] = 1./7
    prior_table[5][11][7] = 1./7
    prior_table[5][11][8] = 1./7
    prior_table[5][11][9] = 0.
    prior_table[5][11][10] = 1./7
    prior_table[5][11][11] = 1./7
    prior_table[5][11][12] = 1./7

    prior_table[6][11][0] = 1./7
    prior_table[6][11][1] = 0.
    prior_table[6][11][2] = 0.
    prior_table[6][11][3] = 0.
    prior_table[6][11][4] = 0.
    prior_table[6][11][5] = 0.
    prior_table[6][11][6] = 1./7
    prior_table[6][11][7] = 1./7
    prior_table[6][11][8] = 1./7
    prior_table[6][11][9] = 0.
    prior_table[6][11][10] = 1./7
    prior_table[6][11][11] = 1./7
    prior_table[6][11][12] = 1./7

    prior_table[7][11][0] = 1./7
    prior_table[7][11][1] = 0.
    prior_table[7][11][2] = 0.
    prior_table[7][11][3] = 0.
    prior_table[7][11][4] = 0.
    prior_table[7][11][5] = 0.
    prior_table[7][11][6] = 1./7
    prior_table[7][11][7] = 1./7
    prior_table[7][11][8] = 1./7
    prior_table[7][11][9] = 0.
    prior_table[7][11][10] = 1./7
    prior_table[7][11][11] = 1./7
    prior_table[7][11][12] = 1./7

    prior_table[8][11][0] = 1./7
    prior_table[8][11][1] = 0.
    prior_table[8][11][2] = 0.
    prior_table[8][11][3] = 0.
    prior_table[8][11][4] = 0.
    prior_table[8][11][5] = 0.
    prior_table[8][11][6] = 1./7
    prior_table[8][11][7] = 1./7
    prior_table[8][11][8] = 1./7
    prior_table[8][11][9] = 0.
    prior_table[8][11][10] = 1./7
    prior_table[8][11][11] = 1./7
    prior_table[8][11][12] = 1./7

    prior_table[9][11][0] = 1./7
    prior_table[9][11][1] = 0.
    prior_table[9][11][2] = 0.
    prior_table[9][11][3] = 0.
    prior_table[9][11][4] = 0.
    prior_table[9][11][5] = 0.
    prior_table[9][11][6] = 1./7
    prior_table[9][11][7] = 1./7
    prior_table[9][11][8] = 1./7
    prior_table[9][11][9] = 0.
    prior_table[9][11][10] = 1./7
    prior_table[9][11][11] = 1./7
    prior_table[9][11][12] = 1./7

    prior_table[10][11][0] = 1./7
    prior_table[10][11][1] = 0.
    prior_table[10][11][2] = 0.
    prior_table[10][11][3] = 0.
    prior_table[10][11][4] = 0.
    prior_table[10][11][5] = 0.
    prior_table[10][11][6] = 1./7
    prior_table[10][11][7] = 1./7
    prior_table[10][11][8] = 1./7
    prior_table[10][11][9] = 0.
    prior_table[10][11][10] = 1./7
    prior_table[10][11][11] = 1./7
    prior_table[10][11][12] = 1./7

    prior_table[11][11][0] = 1./7
    prior_table[11][11][1] = 0.
    prior_table[11][11][2] = 0.
    prior_table[11][11][3] = 0.
    prior_table[11][11][4] = 0.
    prior_table[11][11][5] = 0.
    prior_table[11][11][6] = 1./7
    prior_table[11][11][7] = 1./7
    prior_table[11][11][8] = 1./7
    prior_table[11][11][9] = 0.
    prior_table[11][11][10] = 1./7
    prior_table[11][11][11] = 1./7
    prior_table[11][11][12] = 1./7

    prior_table[12][11][0] = 1./7
    prior_table[12][11][1] = 0.
    prior_table[12][11][2] = 0.
    prior_table[12][11][3] = 0.
    prior_table[12][11][4] = 0.
    prior_table[12][11][5] = 0.
    prior_table[12][11][6] = 1./7
    prior_table[12][11][7] = 1./7
    prior_table[12][11][8] = 1./7
    prior_table[12][11][9] = 0.
    prior_table[12][11][10] = 1./7
    prior_table[12][11][11] = 1./7
    prior_table[12][11][12] = 1./7

    #######################################################

    prior_table[0][12][0] = 1./5
    prior_table[0][12][1] = 0.
    prior_table[0][12][2] = 0.
    prior_table[0][12][3] = 0.
    prior_table[0][12][4] = 0.
    prior_table[0][12][5] = 0.
    prior_table[0][12][6] = 0.
    prior_table[0][12][7] = 1./5
    prior_table[0][12][8] = 1./5
    prior_table[0][12][9] = 0.
    prior_table[0][12][10] = 0.
    prior_table[0][12][11] = 1./5
    prior_table[0][12][12] = 1./5

    prior_table[1][12][0] = 1./5
    prior_table[1][12][1] = 0.
    prior_table[1][12][2] = 0.
    prior_table[1][12][3] = 0.
    prior_table[1][12][4] = 0.
    prior_table[1][12][5] = 0.
    prior_table[1][12][6] = 0.
    prior_table[1][12][7] = 1./5
    prior_table[1][12][8] = 1./5
    prior_table[1][12][9] = 0.
    prior_table[1][12][10] = 0.
    prior_table[1][12][11] = 1./5
    prior_table[1][12][12] = 1./5

    prior_table[2][12][0] = 1./5
    prior_table[2][12][1] = 0.
    prior_table[2][12][2] = 0.
    prior_table[2][12][3] = 0.
    prior_table[2][12][4] = 0.
    prior_table[2][12][5] = 0.
    prior_table[2][12][6] = 0.
    prior_table[2][12][7] = 1./5
    prior_table[2][12][8] = 1./5
    prior_table[2][12][9] = 0.
    prior_table[2][12][10] = 0.
    prior_table[2][12][11] = 1./5
    prior_table[2][12][12] = 1./5

    prior_table[3][12][0] = 1./5
    prior_table[3][12][1] = 0.
    prior_table[3][12][2] = 0.
    prior_table[3][12][3] = 0.
    prior_table[3][12][4] = 0.
    prior_table[3][12][5] = 0.
    prior_table[3][12][6] = 0.
    prior_table[3][12][7] = 1./5
    prior_table[3][12][8] = 1./5
    prior_table[3][12][9] = 0.
    prior_table[3][12][10] = 0.
    prior_table[3][12][11] = 1./5
    prior_table[3][12][12] = 1./5

    prior_table[4][12][0] = 1./5
    prior_table[4][12][1] = 0.
    prior_table[4][12][2] = 0.
    prior_table[4][12][3] = 0.
    prior_table[4][12][4] = 0.
    prior_table[4][12][5] = 0.
    prior_table[4][12][6] = 0.
    prior_table[4][12][7] = 1./5
    prior_table[4][12][8] = 1./5
    prior_table[4][12][9] = 0.
    prior_table[4][12][10] = 0.
    prior_table[4][12][11] = 1./5
    prior_table[4][12][12] = 1./5

    prior_table[5][12][0] = 1./5
    prior_table[5][12][1] = 0.
    prior_table[5][12][2] = 0.
    prior_table[5][12][3] = 0.
    prior_table[5][12][4] = 0.
    prior_table[5][12][5] = 0.
    prior_table[5][12][6] = 0.
    prior_table[5][12][7] = 1./5
    prior_table[5][12][8] = 1./5
    prior_table[5][12][9] = 0.
    prior_table[5][12][10] = 0.
    prior_table[5][12][11] = 1./5
    prior_table[5][12][12] = 1./5

    prior_table[6][12][0] = 1./5
    prior_table[6][12][1] = 0.
    prior_table[6][12][2] = 0.
    prior_table[6][12][3] = 0.
    prior_table[6][12][4] = 0.
    prior_table[6][12][5] = 0.
    prior_table[6][12][6] = 0.
    prior_table[6][12][7] = 1./5
    prior_table[6][12][8] = 1./5
    prior_table[6][12][9] = 0.
    prior_table[6][12][10] = 0.
    prior_table[6][12][11] = 1./5
    prior_table[6][12][12] = 1./5

    prior_table[7][12][0] = 1./5
    prior_table[7][12][1] = 0.
    prior_table[7][12][2] = 0.
    prior_table[7][12][3] = 0.
    prior_table[7][12][4] = 0.
    prior_table[7][12][5] = 0.
    prior_table[7][12][6] = 0.
    prior_table[7][12][7] = 1./5
    prior_table[7][12][8] = 1./5
    prior_table[7][12][9] = 0.
    prior_table[7][12][10] = 0.
    prior_table[7][12][11] = 1./5
    prior_table[7][12][12] = 1./5

    prior_table[8][12][0] = 1./5
    prior_table[8][12][1] = 0.
    prior_table[8][12][2] = 0.
    prior_table[8][12][3] = 0.
    prior_table[8][12][4] = 0.
    prior_table[8][12][5] = 0.
    prior_table[8][12][6] = 0.
    prior_table[8][12][7] = 1./5
    prior_table[8][12][8] = 1./5
    prior_table[8][12][9] = 0.
    prior_table[8][12][10] = 0.
    prior_table[8][12][11] = 1./5
    prior_table[8][12][12] = 1./5

    prior_table[9][12][0] = 1./5
    prior_table[9][12][1] = 0.
    prior_table[9][12][2] = 0.
    prior_table[9][12][3] = 0.
    prior_table[9][12][4] = 0.
    prior_table[9][12][5] = 0.
    prior_table[9][12][6] = 0.
    prior_table[9][12][7] = 1./5
    prior_table[9][12][8] = 1./5
    prior_table[9][12][9] = 0.
    prior_table[9][12][10] = 0.
    prior_table[9][12][11] = 1./5
    prior_table[9][12][12] = 1./5

    prior_table[10][12][0] = 1./5
    prior_table[10][12][1] = 0.
    prior_table[10][12][2] = 0.
    prior_table[10][12][3] = 0.
    prior_table[10][12][4] = 0.
    prior_table[10][12][5] = 0.
    prior_table[10][12][6] = 0.
    prior_table[10][12][7] = 1./5
    prior_table[10][12][8] = 1./5
    prior_table[10][12][9] = 0.
    prior_table[10][12][10] = 0.
    prior_table[10][12][11] = 1./5
    prior_table[10][12][12] = 1./5

    prior_table[11][12][0] = 1./5
    prior_table[11][12][1] = 0.
    prior_table[11][12][2] = 0.
    prior_table[11][12][3] = 0.
    prior_table[11][12][4] = 0.
    prior_table[11][12][5] = 0.
    prior_table[11][12][6] = 0.
    prior_table[11][12][7] = 1./5
    prior_table[11][12][8] = 1./5
    prior_table[11][12][9] = 0.
    prior_table[11][12][10] = 0.
    prior_table[11][12][11] = 1./5
    prior_table[11][12][12] = 1./5

    prior_table[12][12][0] = 1./5
    prior_table[12][12][1] = 0.
    prior_table[12][12][2] = 0.
    prior_table[12][12][3] = 0.
    prior_table[12][12][4] = 0.
    prior_table[12][12][5] = 0.
    prior_table[12][12][6] = 0.
    prior_table[12][12][7] = 1./5
    prior_table[12][12][8] = 1./5
    prior_table[12][12][9] = 0.
    prior_table[12][12][10] = 0.
    prior_table[12][12][11] = 1./5
    prior_table[12][12][12] = 1./5

    ######################################################

    pre_last_state = real_labels[0]
    last_state = real_labels[1]

    #print prior_table
    #print ''

    for label in real_labels[2:]:
        current_state = label
        stat_table[pre_last_state][last_state][current_state] += 1
        pre_last_state = last_state
        last_state = current_state

    for pre_last in np.arange(13):
        for last in np.arange(13):
            sum_prior = np.sum(prior_table[pre_last][last])
            if sum_prior == 0:
                print 'PRIOR SUM ZERO at %d, %d' %(pre_last, last)

    empty_dists = np.zeros((13,13))

    for pre_last in np.arange(13):
        for last in np.arange(13):
            sum_stat = np.sum(stat_table[pre_last][last])
            if sum_stat != 0.:
                stat_table[pre_last][last] /= sum_stat
            if sum_stat == 0:
                empty_dists[pre_last][last] = 1
                print 'STAT SUM ZERO at %d, %d' %(pre_last, last)

    #print stat_table
    #print ''

    for pre_last in np.arange(13):
        for last in np.arange(13):
            if empty_dists[pre_last][last] == 0:
                posterior_table[pre_last][last] = prior_table[pre_last][last] * stat_table[pre_last][last]
            else:
                posterior_table[pre_last][last] = prior_table[pre_last][last]

    for pre_last in np.arange(13):
        for last in np.arange(13):
            sum_post = np.sum(posterior_table[pre_last][last])
            if sum_post != 0.:
                posterior_table[pre_last][last] /= sum_post
            else:
                print 'SUM ZERO at %d, %d' %(pre_last, last)

    posterior_table[0][0][0] = 1./11
    posterior_table[0][0][1] = 1./11
    posterior_table[0][0][2] = 1./11
    posterior_table[0][0][3] = 1./11
    posterior_table[0][0][4] = 1./11
    posterior_table[0][0][5] = 1./11
    posterior_table[0][0][6] = 0.
    posterior_table[0][0][7] = 0.
    posterior_table[0][0][8] = 1./11
    posterior_table[0][0][9] = 1./11
    posterior_table[0][0][10] = 1./11
    posterior_table[0][0][11] = 1./11
    posterior_table[0][0][12] = 1./11

    posterior_table[1][0][0] = 1./11
    posterior_table[1][0][1] = 1./11
    posterior_table[1][0][2] = 1./11
    posterior_table[1][0][3] = 1./11
    posterior_table[1][0][4] = 1./11
    posterior_table[1][0][5] = 1./11
    posterior_table[1][0][6] = 0.
    posterior_table[1][0][7] = 0.
    posterior_table[1][0][8] = 1./11
    posterior_table[1][0][9] = 1./11
    posterior_table[1][0][10] = 1./11
    posterior_table[1][0][11] = 1./11
    posterior_table[1][0][12] = 1./11

    posterior_table[2][0][0] = 1./11
    posterior_table[2][0][1] = 1./11
    posterior_table[2][0][2] = 1./11
    posterior_table[2][0][3] = 1./11
    posterior_table[2][0][4] = 1./11
    posterior_table[2][0][5] = 1./11
    posterior_table[2][0][6] = 0.
    posterior_table[2][0][7] = 0.
    posterior_table[2][0][8] = 1./11
    posterior_table[2][0][9] = 1./11
    posterior_table[2][0][10] = 1./11
    posterior_table[2][0][11] = 1./11
    posterior_table[2][0][12] = 1./11

    posterior_table[3][0][0] = 1./11
    posterior_table[3][0][1] = 1./11
    posterior_table[3][0][2] = 1./11
    posterior_table[3][0][3] = 1./11
    posterior_table[3][0][4] = 1./11
    posterior_table[3][0][5] = 1./11
    posterior_table[3][0][6] = 0.
    posterior_table[3][0][7] = 0.
    posterior_table[3][0][8] = 1./11
    posterior_table[3][0][9] = 1./11
    posterior_table[3][0][10] = 1./11
    posterior_table[3][0][11] = 1./11
    posterior_table[3][0][12] = 1./11

    posterior_table[4][0][0] = 1./11
    posterior_table[4][0][1] = 1./11
    posterior_table[4][0][2] = 1./11
    posterior_table[4][0][3] = 1./11
    posterior_table[4][0][4] = 1./11
    posterior_table[4][0][5] = 1./11
    posterior_table[4][0][6] = 0.
    posterior_table[4][0][7] = 0.
    posterior_table[4][0][8] = 1./11
    posterior_table[4][0][9] = 1./11
    posterior_table[4][0][10] = 1./11
    posterior_table[4][0][11] = 1./11
    posterior_table[4][0][12] = 1./11

    posterior_table[5][0][0] = 1./11
    posterior_table[5][0][1] = 1./11
    posterior_table[5][0][2] = 1./11
    posterior_table[5][0][3] = 1./11
    posterior_table[5][0][4] = 1./11
    posterior_table[5][0][5] = 1./11
    posterior_table[5][0][6] = 0.
    posterior_table[5][0][7] = 0.
    posterior_table[5][0][8] = 1./11
    posterior_table[5][0][9] = 1./11
    posterior_table[5][0][10] = 1./11
    posterior_table[5][0][11] = 1./11
    posterior_table[5][0][12] = 1./11

    posterior_table[6][0][0] = 1./11
    posterior_table[6][0][1] = 1./11
    posterior_table[6][0][2] = 1./11
    posterior_table[6][0][3] = 1./11
    posterior_table[6][0][4] = 1./11
    posterior_table[6][0][5] = 1./11
    posterior_table[6][0][6] = 0.
    posterior_table[6][0][7] = 0.
    posterior_table[6][0][8] = 1./11
    posterior_table[6][0][9] = 1./11
    posterior_table[6][0][10] = 1./11
    posterior_table[6][0][11] = 1./11
    posterior_table[6][0][12] = 1./11

    posterior_table[7][0][0] = 1./11
    posterior_table[7][0][1] = 1./11
    posterior_table[7][0][2] = 1./11
    posterior_table[7][0][3] = 1./11
    posterior_table[7][0][4] = 1./11
    posterior_table[7][0][5] = 1./11
    posterior_table[7][0][6] = 0.
    posterior_table[7][0][7] = 0.
    posterior_table[7][0][8] = 1./11
    posterior_table[7][0][9] = 1./11
    posterior_table[7][0][10] = 1./11
    posterior_table[7][0][11] = 1./11
    posterior_table[7][0][12] = 1./11

    posterior_table[8][0][0] = 1./11
    posterior_table[8][0][1] = 1./11
    posterior_table[8][0][2] = 1./11
    posterior_table[8][0][3] = 1./11
    posterior_table[8][0][4] = 1./11
    posterior_table[8][0][5] = 1./11
    posterior_table[8][0][6] = 0.
    posterior_table[8][0][7] = 0.
    posterior_table[8][0][8] = 1./11
    posterior_table[8][0][9] = 1./11
    posterior_table[8][0][10] = 1./11
    posterior_table[8][0][11] = 1./11
    posterior_table[8][0][12] = 1./11

    posterior_table[9][0][0] = 1./11
    posterior_table[9][0][1] = 1./11
    posterior_table[9][0][2] = 1./11
    posterior_table[9][0][3] = 1./11
    posterior_table[9][0][4] = 1./11
    posterior_table[9][0][5] = 1./11
    posterior_table[9][0][6] = 0.
    posterior_table[9][0][7] = 0.
    posterior_table[9][0][8] = 1./11
    posterior_table[9][0][9] = 1./11
    posterior_table[9][0][10] = 1./11
    posterior_table[9][0][11] = 1./11
    posterior_table[9][0][12] = 1./11

    posterior_table[10][0][0] = 1./11
    posterior_table[10][0][1] = 1./11
    posterior_table[10][0][2] = 1./11
    posterior_table[10][0][3] = 1./11
    posterior_table[10][0][4] = 1./11
    posterior_table[10][0][5] = 1./11
    posterior_table[10][0][6] = 0.
    posterior_table[10][0][7] = 0.
    posterior_table[10][0][8] = 1./11
    posterior_table[10][0][9] = 1./11
    posterior_table[10][0][10] = 1./11
    posterior_table[10][0][11] = 1./11
    posterior_table[10][0][12] = 1./11

    posterior_table[11][0][0] = 1./11
    posterior_table[11][0][1] = 1./11
    posterior_table[11][0][2] = 1./11
    posterior_table[11][0][3] = 1./11
    posterior_table[11][0][4] = 1./11
    posterior_table[11][0][5] = 1./11
    posterior_table[11][0][6] = 0.
    posterior_table[11][0][7] = 0.
    posterior_table[11][0][8] = 1./11
    posterior_table[11][0][9] = 1./11
    posterior_table[11][0][10] = 1./11
    posterior_table[11][0][11] = 1./11
    posterior_table[11][0][12] = 1./11

    posterior_table[12][0][0] = 1./11
    posterior_table[12][0][1] = 1./11
    posterior_table[12][0][2] = 1./11
    posterior_table[12][0][3] = 1./11
    posterior_table[12][0][4] = 1./11
    posterior_table[12][0][5] = 1./11
    posterior_table[12][0][6] = 0.
    posterior_table[12][0][7] = 0.
    posterior_table[12][0][8] = 1./11
    posterior_table[12][0][9] = 1./11
    posterior_table[12][0][10] = 1./11
    posterior_table[12][0][11] = 1./11
    posterior_table[12][0][12] = 1./11


    #print posterior_table
    #print ''

    for i in np.arange(13):
        for j in np.arange(13):
            for k in np.arange(13):
                if posterior_table[i][j][k] == np.nan:
                    print 'NAN'


    cPickle.dump(posterior_table, open('posterior_table_2nd.pkl', 'wb'))


if __name__ == '__main__':
    compute_markov_table()