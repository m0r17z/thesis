
import numpy as np
import h5py as h5
from utils import determine_label
from generate_datasets import generate_train_val_test_set

def generate_real_dataset_binning_cnn():
    ################################################ LOADING AND CLEANING THE DATA #########################################
    samples = open('./samples.txt')
    labels = open('./labels.txt')
    annotations = open('./annotations.txt')

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
    bin_cm = 10
    max_x_cm = 440
    min_x_cm = 70
    max_y_cm = 250
    max_z_cm = 200
    nr_z_intervals = 2

    x_range = max_x_cm/bin_cm - min_x_cm/bin_cm
    y_range = max_y_cm*2/bin_cm
    z_range = nr_z_intervals

    f = h5.File("./usarray_data_unscaled_real_cnn.hdf5", "w")
    f.create_dataset('data_set/data_set', (len(qpoint_lists),z_range, x_range*y_range), dtype='f')
    f.create_dataset('labels/real_labels', (len(real_labels),), dtype='i')
    dt = h5.special_dtype(vlen=unicode)
    f.create_dataset('annotations/annotations', (len(annotation_list),), dtype=dt)

    last_per = -1

    for ind, qpoint_list in enumerate(qpoint_lists):
        grid = np.zeros((z_range, x_range, y_range))

        for qpoint in qpoint_list:
            x = int(float(qpoint[0])*100) / bin_cm
            y = (int(float(qpoint[1])*100) + max_y_cm) / bin_cm
            # this actually only works if z_range == 2
            z = int(float(qpoint[2])*100) > (max_z_cm / nr_z_intervals)
            if x < min_x_cm/bin_cm or x > max_x_cm/bin_cm-1 or y > max_y_cm*2/bin_cm-1 or y < 0:
                continue
            pow = float(qpoint[4])
            if grid[z][x-min_x_cm/bin_cm][y] != 0:
                pcol += 1
                if grid[z][x-min_x_cm/bin_cm][y] < pow:
                    grid[z][x-min_x_cm/bin_cm][y] = pow
            grid[z][x-min_x_cm/bin_cm][y] = pow
            ps += 1

        # unroll the grid into a vector?!
        for z_val in np.arange(z_range):
            f['data_set/data_set'][ind][z_val] = grid[z_val].flatten()
        f['labels/real_labels'][ind] = real_labels[ind]
        f['annotations/annotations'][ind] = annotation_list[ind]
        curr_percent = int(float(ind) / len(qpoint_lists) * 100)
        if last_per != curr_percent:
            last_per = curr_percent
            print 'have now looked at %i%% of the data.' % int(float(ind) / len(qpoint_lists) * 100)

    print 'percentage of point collision: ' + str(float(pcol)/ps)
    print 'number of samples: ' +str(len(f['data_set/data_set']))
    print 'dimensionality of the samples: ' +str(len(f['data_set/data_set'][0]))
    print 'number of labels: ' +str(len(f['labels/real_labels']))
    print 'number of annotations: ' +str(len(f['annotations/annotations']))

    f.close()

    generate_train_val_test_set("./usarray_data_unscaled_binning_real_cnn.hdf5", "usarray_data_unscaled_train_val_test_binning_real_cnn.hdf5")
