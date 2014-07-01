
import numpy as np
import h5py as h5
from generate_datasets import generate_train_val_test_set

############################## WAS ONLY USED FOR INITIAL TESTS - IS NOT UP 2 DATE ##########################################
def generate_bin_dataset_binning():
    ################################################ LOADING AND CLEANING THE DATA #########################################
    samples = open('/local-home/moritz/Dataset/samples.txt')
    labels = open('/local-home/moritz/Dataset/labels.txt')
    annotations = open('/local-home/moritz/Dataset/annotations.txt')

    bad_samples = []
    bin_labels = []
    qpoint_lists = []
    label_list = []
    annotation_list = []

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
        ################# PROCESS THE LABELS
        if annotation_list[list_ind][0:2] != 'vo' and annotation_list[list_ind][0:2] != 'fl' and annotation_list[list_ind][0:2] != 'mi' and annotation_list[list_ind][0:2] != 'ja':
            bin_labels.append(0)
        else:
            bin_labels.append(1)
        ################# PROCESS THE Q-POINTS
        qpoint_lists[list_ind] = qpoint_lists[list_ind].split(':')
        #print 'found the %ith list to have %i qpoints.' %(list_ind+1, len(qpoint_lists[list_ind]) )
        for point_ind in np.arange(len(qpoint_lists[list_ind])):
            qpoint_lists[list_ind][point_ind] = qpoint_lists[list_ind][point_ind].split(',')
            if len(qpoint_lists[list_ind][point_ind]) != 7:
                #print 'WARNING: QPoint found to have not enough values.'
                #print 'WARNING: QPoint values: %s' % str(qpoint_lists[list_ind][point_ind])
                #print 'WARNING: Length of list is: %i' % len(qpoint_lists[list_ind])
                bad_samples.append(list_ind)
            else:
                pass
                #print qpoint_lists[list_ind][point_ind]
            #print 'found the %ith qpoint in the %ith list to have %i values.' %( point_ind+1,list_ind+1 ,
            #  len(qpoint_lists[list_ind][point_ind]) )

    print 'need to remove %i empty samples.' %len(bad_samples)
    ################# REMOVE EMPTY SAMPLES
    for ind, bad_ind in enumerate(bad_samples):
        real_ind = bad_ind - ind
        #print 'sample at position %i has %i qpoints.' %(real_ind, len(qpoint_lists[real_ind]))
        #print 'removing that one.'
        qpoint_lists.pop(real_ind)
        bin_labels.pop(real_ind)
        annotation_list.pop(real_ind)

    print str(len(qpoint_lists)) + ' samples remain after purging.'
    print str(len(bin_labels)) + ' labels remain after purging.'
    print str(len(annotation_list)) + ' annotations remain after purging.'
    samples.close()
    labels.close()
    annotations.close()

    ################################################## PROJECTING THE DATA INTO A GRID #####################################
    pcol = 0
    ps = 0

    # ASSUMPTION: relevant area is never less than 0.9 meters and more than 4 meters on the x-axis, 2.2 meters to both sides on the y-axis
    # and 2 meters on the z-axis away from the sensors
    # unit is 2 centimeters here
    bin_cm = 10
    max_x_cm = 400
    min_x_cm = 90
    max_y_cm = 220
    max_z_cm = 200
    nr_z_intervals = 2

    x_range = max_x_cm/bin_cm - min_x_cm/bin_cm
    y_range = max_y_cm*2/bin_cm
    z_range = nr_z_intervals

    f = h5.File("usarray_data_unscaled_bin.hdf5", "w")
    f.create_dataset('data_set/data_set', (len(qpoint_lists),x_range*y_range*z_range), dtype='f')
    f.create_dataset('labels/bin_labels', (len(bin_labels),), dtype='i')
    dt = h5.special_dtype(vlen=unicode)
    f.create_dataset('annotations/annotations', (len(annotation_list),), dtype=dt)

    last_per = -1

    for i,qpoint_list in enumerate(qpoint_lists):
        grid = np.zeros((x_range, y_range, z_range))
        #print grid.shape
        for qpoint in qpoint_list:
            x = int(float(qpoint[0])*100) / bin_cm
            y = (int(float(qpoint[1])*100) + max_y_cm) / bin_cm
            z = int(float(qpoint[2])*100) > (max_z_cm / nr_z_intervals)
            if x < min_x_cm/bin_cm or x > max_x_cm/bin_cm-1 or y > max_y_cm*2/bin_cm-1 or y < 0:# or z < 0 or z > 199:
                #print 'found QPoint out of range: ignoring it.'
                continue
            pow = float(qpoint[4])
            #print 'QPoint %s is inserted into the grid at position (%i,%i,%i) with value %f ' %(qpoint,x,y,z,pow)
            if grid[x-min_x_cm/bin_cm][y][z] != 0:
                #print 'WARNING: Point Collision.'
                pcol += 1
                if grid[x-min_x_cm/bin_cm][y][z] < pow:
                    grid[x-min_x_cm/bin_cm][y][z] = pow
            else:
                grid[x-min_x_cm/bin_cm][y][z] = pow
            ps += 1

        # unroll the grid into a vector?!
        f['data_set/data_set'][i] = grid.flatten()
        f['labels/bin_labels'][i] = bin_labels[i]
        f['annotations/annotations'][i] = annotation_list[i]
        curr_percent = int(float(i) / len(qpoint_lists) * 100)
        if last_per != curr_percent:
            last_per = curr_percent
            print 'have now looked at %i%% of the data.' % int(float(i) / len(qpoint_lists) * 100)

    print 'percentage of point collision: ' + str(float(pcol)/ps)
    print 'number of samples: ' +str(len(f['data_set/data_set']))
    print 'dimensionality of the samples: ' +str(len(f['data_set/data_set'][0]))
    print 'number of labels: ' +str(len(f['labels/bin_labels']))
    print 'number of annotations: ' +str(len(f['annotations/annotations']))

    f.close()