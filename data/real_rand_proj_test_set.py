
"""Usage:
    real_rand_proj_data_int.py <path> <sparse> <eps>

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
    labels = open(os.path.join(data_path, 'labels.txt'))
    annotations = open(os.path.join(data_path, 'annotations.txt'))
    out_f = open(os.path.join(data_path,'rp_out_test'),'w')

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

    out_s = 'found %i qpoint lists.\n' % len(qpoint_lists) + 'found %i labels.\n' % len(label_list) + 'found %i annotations.\n\n' % len(annotation_list)
    print out_s
    out_f.write(out_s)
    out_f.close()

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

    out_f = open(os.path.join(data_path,'rp_out_test'),'a')
    out_s = str(len(qpoint_lists)) + ' samples remain after purging.\n' + str(len(real_labels)) + ' labels remain after purging.\n'\
            + str(len(annotation_list)) + ' annotations remain after purging.\n' + 'percentages of the labels are %s\n\n' %str(label_count/len(qpoint_lists))
    print out_s
    out_f.write(out_s)
    out_f.close()

    samples.close()
    labels.close()
    annotations.close()

    ################################################## PROJECTING THE DATA INTO A GRID #####################################
    pcol = 0
    ps = 0

    # ASSUMPTION: relevant area is never less than 0.7 meters and more than 4.4 meters on the x-axis, 2.5 meters to both sides on the y-axis
    # and 2 meters on the z-axis away from the sensors
    bin_cm = 3
    max_x_cm = 440
    min_x_cm = 70
    max_y_cm = 250
    max_z_cm = 200

    x_range = max_x_cm / bin_cm - min_x_cm / bin_cm
    y_range = max_y_cm * 2 / bin_cm
    z_range = max_z_cm / bin_cm

    out_f = open(os.path.join(data_path,'rp_out_test'),'a')
    out_s = 'length of data in original space: %d\n\n' %(x_range*y_range*z_range)
    print out_s
    out_f.write(out_s)
    out_f.close()

    # compute a conservative estimate of the number of latent dimensions required to guarantuee the given epsilons
    n_dims = johnson_lindenstrauss_min_dim(len(qpoint_lists),eps)

    out_f = open(os.path.join(data_path,'rp_out_test'),'a')
    out_s = 'number of latent dimensions needed to guarantee %f epsilon is %f\n\n' %(eps, n_dims)
    print out_s
    out_f.write(out_s)
    out_f.close()

    f_path = os.path.join(data_path,'rp_real_sparse.hdf5') if sparse else os.path.join(data_path,'rp_real_gauss.hdf5')
    print f_path
    f = h5.File(f_path, "w")
    f.create_dataset('test_set/test_set', (90000, n_dims), dtype='f')
    f.create_dataset('test_labels/real_test_labels', (90000,), dtype='i')
    dt = h5.special_dtype(vlen=unicode)
    f.create_dataset('test_annotations/test_annotations', (90000,), dtype=dt)

    transformer = random_projection.SparseRandomProjection(n_components=n_dims) if sparse else random_projection.GaussianRandomProjection(n_components=n_dims)
    if sparse:
        print 'performing projection with sparse matrix'
    else:
        print 'performing projection with gaussian matrix'

    # this is not the way it's supposed to be done BUT the proper training set doesn't fit into the memory
    transformer.components_ = transformer._make_random_matrix(n_dims, x_range*y_range*z_range)
    last_per = -1
    nr_samples = 0

    for ind, qpoint_list in enumerate(qpoint_lists):

        if nr_samples > 90000:
            break

        rand = np.random.random()
        if rand > 0.25:
            continue

        nr_samples += 1

        grid = np.zeros((x_range, y_range, z_range))

        for qpoint in qpoint_list:
            x = int(float(qpoint[0])*100) / bin_cm
            y = (int(float(qpoint[1])*100) + max_y_cm) / bin_cm
            z = int(float(qpoint[2])*100) / bin_cm
            if x - min_x_cm/bin_cm < 0 or x - min_x_cm/bin_cm > x_range-1 or y > y_range-1 or y < 0 or z > z_range-1 or z < 0:
                continue
            pow = float(qpoint[4])
            if grid[x-min_x_cm/bin_cm][y][z] != 0:
                pcol += 1
                if grid[x-min_x_cm/bin_cm][y][z] < pow:
                    grid[x-min_x_cm/bin_cm][y][z] = pow
            else:
                grid[x-min_x_cm/bin_cm][y][z] = pow
            ps += 1

        if not nr_samples > 90000:
            f['test_set/test_set'][nr_samples-1] = transformer.transform(np.reshape(grid,(1,-1)))
            f['test_labels/real_test_abels'][nr_samples-1] = real_labels[ind]
            f['test_annotations/test_annotations'][nr_samples-1] = annotation_list[ind]
            curr_percent = int(float(nr_samples) / 90000. * 100)
            if last_per != curr_percent:
                last_per = curr_percent
                out_f = open(os.path.join(data_path,'rp_out_test'),'a')
                out_s = 'have now looked at %i%% of the data.\n' % int(float(ind) / len(qpoint_lists) * 100)
                print out_s
                out_f.write(out_s)
                out_f.close()

    print 'done with projecting onto the grid (without binning)'
    print 'percentage of point collision: ' + str(float(pcol)/ps)
    print 'number of samples: ' +str(len(f['test_set/test_set']))
    print 'dimensionality of the samples: ' +str(len(f['test_set/test_set'][0]))
    print 'number of labels: ' +str(len(f['test_labels/real_test_labels']))
    print 'number of annotations: ' +str(len(f['test_annotations/test_annotations']))

    out_f = open(os.path.join(data_path,'rp_out_test'),'a')
    out_s = 'projection done, new dimension is %d\n\n' %len(f['test_set/test_set'][0])
    print out_s
    out_f.write(out_s)
    out_f.close()

    f.close()

if __name__ == '__main__':
    #args = docopt.docopt(__doc__)
    #print args
    #eps = float(args['<eps>'])
    #sparse = args['<sparse>']
    #path = args['<path>']
    generate_real_dataset_rp('/nthome/maugust/thesis', True, 0.3)
