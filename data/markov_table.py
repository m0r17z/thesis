import numpy as np
import cPickle

from utils import determine_label

def compute_markov_table():
    samples = open('/nthome/maugust/thesis/samples_int_ordered.txt')
    labels = open('/nthome/maugust/thesis/labels_int_ordered.txt')
    annotations = open('/nthome/maugust/thesis/annotations_int_ordered.txt')

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

    prior_table = np.zeros((13, 13), dtype=np.float32)
    stat_table = np.zeros((13, 13), dtype=np.float32)

    prior_table[0][0] = 1/11
    prior_table[0][1] = 1/11
    prior_table[0][2] = 1/11
    prior_table[0][3] = 1/11
    prior_table[0][4] = 1/11
    prior_table[0][5] = 1/11
    prior_table[0][6] = 0
    prior_table[0][7] = 0
    prior_table[0][8] = 1/11
    prior_table[0][9] = 1/11
    prior_table[0][10] = 1/11
    prior_table[0][11] = 1/11
    prior_table[0][12] = 1/11

    prior_table[1][0] = 1/5
    prior_table[1][1] = 1/5
    prior_table[1][2] = 1/5
    prior_table[1][3] = 0
    prior_table[1][4] = 0
    prior_table[1][5] = 1/5
    prior_table[1][6] = 1/5
    prior_table[1][7] = 0
    prior_table[1][8] = 0
    prior_table[1][9] = 0
    prior_table[1][10] = 0
    prior_table[1][11] = 0
    prior_table[1][12] = 0

    prior_table[2][0] = 1/7
    prior_table[2][1] = 1/7
    prior_table[2][2] = 1/7
    prior_table[2][3] = 1/7
    prior_table[2][4] = 0
    prior_table[2][5] = 1/7
    prior_table[2][6] = 1/7
    prior_table[2][7] = 1/7
    prior_table[2][8] = 0
    prior_table[2][9] = 0
    prior_table[2][10] = 0
    prior_table[2][11] = 0
    prior_table[2][12] = 0

    prior_table[3][0] = 1/7
    prior_table[3][1] = 0
    prior_table[3][2] = 1/7
    prior_table[3][3] = 1/7
    prior_table[3][4] = 1/7
    prior_table[3][5] = 0
    prior_table[3][6] = 1/7
    prior_table[3][7] = 1/7
    prior_table[3][8] = 1/7
    prior_table[3][9] = 0
    prior_table[3][10] = 0
    prior_table[3][11] = 0
    prior_table[3][12] = 0

    prior_table[4][0] = 1/5
    prior_table[4][1] = 0
    prior_table[4][2] = 0
    prior_table[4][3] = 1/5
    prior_table[4][4] = 1/5
    prior_table[4][5] = 0
    prior_table[4][6] = 0
    prior_table[4][7] = 1/5
    prior_table[4][8] = 1/5
    prior_table[4][9] = 0
    prior_table[4][10] = 0
    prior_table[4][11] = 0
    prior_table[4][12] = 0

    prior_table[5][0] = 1/7
    prior_table[5][1] = 1/7
    prior_table[5][2] = 1/7
    prior_table[5][3] = 0
    prior_table[5][4] = 0
    prior_table[5][5] = 1/7
    prior_table[5][6] = 1/7
    prior_table[5][7] = 0
    prior_table[5][8] = 0
    prior_table[5][9] = 1/7
    prior_table[5][10] = 1/7
    prior_table[5][11] = 0
    prior_table[5][12] = 0

    prior_table[6][0] = 0
    prior_table[6][1] = 1/9
    prior_table[6][2] = 1/9
    prior_table[6][3] = 1/9
    prior_table[6][4] = 0
    prior_table[6][5] = 1/9
    prior_table[6][6] = 1/9
    prior_table[6][7] = 1/9
    prior_table[6][8] = 0
    prior_table[6][9] = 1/9
    prior_table[6][10] = 1/9
    prior_table[6][11] = 1/9
    prior_table[6][12] = 0

    prior_table[7][0] = 0
    prior_table[7][1] = 0
    prior_table[7][2] = 1/9
    prior_table[7][3] = 1/9
    prior_table[7][4] = 1/9
    prior_table[7][5] = 0
    prior_table[7][6] = 1/9
    prior_table[7][7] = 1/9
    prior_table[7][8] = 1/9
    prior_table[7][9] = 0
    prior_table[7][10] = 1/9
    prior_table[7][11] = 1/9
    prior_table[7][12] = 1/9

    prior_table[8][0] = 1/7
    prior_table[8][1] = 0
    prior_table[8][2] = 0
    prior_table[8][3] = 1/7
    prior_table[8][4] = 1/7
    prior_table[8][5] = 0
    prior_table[8][6] = 0
    prior_table[8][7] = 1/7
    prior_table[8][8] = 1/7
    prior_table[8][9] = 0
    prior_table[8][10] = 0
    prior_table[8][11] = 1/7
    prior_table[8][12] = 1/7

    prior_table[9][0] = 1/5
    prior_table[9][1] = 0
    prior_table[9][2] = 0
    prior_table[9][3] = 0
    prior_table[9][4] = 0
    prior_table[9][5] = 1/5
    prior_table[9][6] = 1/5
    prior_table[9][7] = 0
    prior_table[9][8] = 0
    prior_table[9][9] = 1/5
    prior_table[9][10] = 1/5
    prior_table[9][11] = 0
    prior_table[9][12] = 0

    prior_table[10][0] = 1/7
    prior_table[10][1] = 0
    prior_table[10][2] = 0
    prior_table[10][3] = 0
    prior_table[10][4] = 0
    prior_table[10][5] = 1/7
    prior_table[10][6] = 1/7
    prior_table[10][7] = 1/7
    prior_table[10][8] = 0
    prior_table[10][9] = 1/7
    prior_table[10][10] = 1/7
    prior_table[10][11] = 1/7
    prior_table[10][12] = 0

    prior_table[11][0] = 1/7
    prior_table[11][1] = 0
    prior_table[11][2] = 0
    prior_table[11][3] = 0
    prior_table[11][4] = 0
    prior_table[11][5] = 0
    prior_table[11][6] = 1/7
    prior_table[11][7] = 1/7
    prior_table[11][8] = 1/7
    prior_table[11][9] = 0
    prior_table[11][10] = 1/7
    prior_table[11][11] = 1/7
    prior_table[11][12] = 1/7

    prior_table[12][0] = 1/5
    prior_table[12][1] = 0
    prior_table[12][2] = 0
    prior_table[12][3] = 0
    prior_table[12][4] = 0
    prior_table[12][5] = 0
    prior_table[12][6] = 0
    prior_table[12][7] = 1/5
    prior_table[12][8] = 1/5
    prior_table[12][9] = 0
    prior_table[12][10] = 0
    prior_table[12][11] = 1/5
    prior_table[12][12] = 1/5

    last_state = real_labels[0]

    for label in real_labels[1:]:
        current_state = label
        stat_table[last_state][current_state] += 1
        last_state = current_state

    for row in np.arange(13):
        stat_table[row] /= np.sum(stat_table[row], axis=1)

    posterior_table = prior_table * stat_table

    for row in np.arange(13):
        posterior_table[row] /= np.sum(posterior_table[row], axis=1)

    cPickle.dump(posterior_table, open('posterior_table.pkl', 'wb'))