
"""Usage:
    real_rand_proj_data.py <path> <sparse> <eps>

"""


import numpy as np
import h5py as h5
import os
from utils import determine_label
from generate_datasets import generate_train_val_test_set
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn import random_projection

import docopt


def generate_real_dataset_rp(data_path, sparse=False, eps=0.1):
    ################################################ LOADING AND CLEANING THE DATA #########################################
    samples = open(os.path.join(data_path, 'samples.txt'))
    labels = open(os.path.join(data_path, './labels.txt'))
    annotations = open(os.path.join(data_path, './annotations.txt'))

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

    ################################################## PROJECTING THE DATA INTO A GRID #####################################
    pcol = 0
    ps = 0

    # ASSUMPTION: relevant area is never less than 0.7 meters and more than 4.4 meters on the x-axis, 2.5 meters to both sides on the y-axis
    # and 2 meters on the z-axis away from the sensors
    bin_cm = 1
    max_x_cm = 440
    min_x_cm = 70
    max_y_cm = 250
    max_z_cm = 200

    x_range = max_x_cm / bin_cm - min_x_cm / bin_cm
    y_range = max_y_cm * 2 / bin_cm
    z_range = max_z_cm / bin_cm

    print 'length of data in original space: %d' %(x_range*y_range*z_range)

    # compute a conservative estimate of the number of latent dimensions required to guarantuee the given epsilons
    n_dims = johnson_lindenstrauss_min_dim(len(qpoint_lists),eps)
    print 'number of latent dimensions needed to guarantee %f epsilon is %f' %(eps, n_dims)

    f_path = os.path.join(data_path,'rp_real_sparse.hdf5') if sparse else f_path = os.path.join(data_path,'rp_real_gauss.hdf5')
    print f_path
    f = h5.File(f_path, "w")
    f.create_dataset('data_set/data_set', (len(qpoint_lists), n_dims), dtype='f')
    f.create_dataset('labels/real_labels', (len(real_labels),), dtype='i')
    dt = h5.special_dtype(vlen=unicode)
    f.create_dataset('annotations/annotations', (len(annotation_list),), dtype=dt)

    transformer = random_projection.SparseRandomProjection(n_components=n_dims) if sparse else random_projection.GaussianRandomProjection(n_components=n_dims)
    if sparse:
        print 'performing projection with sparse matrix'
    else:
        print 'performing projection with gaussian matrix'

    # this is not the way it's supposed to be done BUT the proper training set doesn't fit into the memory
    transformer.components_ = transformer._make_random_matrix(n_dims, x_range*y_range*z_range)
    last_per = -1

    for ind, qpoint_list in enumerate(qpoint_lists):
        grid = np.zeros((x_range, y_range, z_range))

        for qpoint in qpoint_list:
            x = int(float(qpoint[0])*100) / bin_cm
            y = (int(float(qpoint[1])*100) + max_y_cm) / bin_cm
            z = int(float(qpoint[2])*100) / bin_cm
            if x < min_x_cm/bin_cm or x > max_x_cm/bin_cm-1 or y > max_y_cm*2/bin_cm-1 or y < 0 or z > max_z_cm-1 or z < 0:
                continue
            pow = float(qpoint[4])
            if grid[x-min_x_cm/bin_cm][y][z] != 0:
                pcol += 1
                if grid[x-min_x_cm/bin_cm][y][z] < pow:
                    grid[x-min_x_cm/bin_cm][y][z] = pow
            grid[x-min_x_cm/bin_cm][y][z] = pow
            ps += 1

        f['data_set/data_set'][ind] = transformer.transform(np.reshape(grid,(1,-1)))
        f['labels/real_labels'][ind] = real_labels[ind]
        f['annotations/annotations'][ind] = annotation_list[ind]
        curr_percent = int(float(ind) / len(qpoint_lists) * 100)
        if last_per != curr_percent:
            last_per = curr_percent
            print 'have now looked at %i%% of the data.' % int(float(ind) / len(qpoint_lists) * 100)

    print 'done with projecting onto the grid (without binning)'
    print 'percentage of point collision: ' + str(float(pcol)/ps)
    print 'number of samples: ' +str(len(f['data_set/data_set']))
    print 'dimensionality of the samples: ' +str(len(f['data_set/data_set'][0]))
    print 'number of labels: ' +str(len(f['labels/real_labels']))
    print 'number of annotations: ' +str(len(f['annotations/annotations']))


    print 'projection done, new dimension is %d' %len(f['data_set/data_set'][0])

    f.close()

    generate_train_val_test_set(os.path.join(data,"usarray_data_rp_real.hdf5"), os.path.join(data,"usarray_data_train_val_test_rp_real.hdf5"))


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    print args
    eps = float(args['<eps>'])
    sparse = args['<sparse>']
    path = args['<path>']
    generate_real_dataset_rp(path, sparse, eps)
