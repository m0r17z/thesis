__author__ = 'moritz'

import h5py as h5
import numpy as np
from sklearn.preprocessing import scale
import warnings


def determine_label((x,y,z)):
    if  x > 3:
        if y > 1:
            return 1
        elif y <= 1 and y > 0:
            return 2
        elif y <= 0 and y > -1:
            return 3
        elif y <= -1:
            return 4
    elif x <= 3 and x > 2:
        if y > 1:
            return 5
        elif y <= 1 and y > 0:
            return 6
        elif y <= 0 and y > -1:
            return 7
        elif y <= -1:
            return 8
    elif x <= 2:
        if y > 1:
            return 9
        elif y <= 1 and y > 0:
            return 10
        elif y <= 0 and y > -1:
            return 11
        elif y <= -1:
            return 12

def generate_real_dataset_binning():
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

    f = h5.File("./usarray_data_unscaled_real.hdf5", "w")
    f.create_dataset('data_set/data_set', (len(qpoint_lists),x_range*y_range*z_range), dtype='f')
    f.create_dataset('labels/real_labels', (len(real_labels),), dtype='i')
    dt = h5.special_dtype(vlen=unicode)
    f.create_dataset('annotations/annotations', (len(annotation_list),), dtype=dt)

    last_per = -1

    for i,qpoint_list in enumerate(qpoint_lists):
        grid = np.zeros((x_range, y_range, z_range))

        for qpoint in qpoint_list:
            x = int(float(qpoint[0])*100) / bin_cm
            y = (int(float(qpoint[1])*100) + max_y_cm) / bin_cm
            z = int(float(qpoint[2])*100) > (max_z_cm / nr_z_intervals)
            if x < min_x_cm/bin_cm or x > max_x_cm/bin_cm-1 or y > max_y_cm*2/bin_cm-1 or y < 0:
                continue
            pow = float(qpoint[4])
            if grid[x-min_x_cm/bin_cm][y][z] != 0:
                pcol += 1
                if grid[x-min_x_cm/bin_cm][y][z] < pow:
                    grid[x-min_x_cm/bin_cm][y][z] = pow
            grid[x-min_x_cm/bin_cm][y][z] = pow
            ps += 1

        # unroll the grid into a vector?!
        f['data_set/data_set'][i] = grid.flatten()
        f['labels/real_labels'][i] = real_labels[i]
        f['annotations/annotations'][i] = annotation_list[i]
        curr_percent = int(float(i) / len(qpoint_lists) * 100)
        if last_per != curr_percent:
            last_per = curr_percent
            print 'have now looked at %i%% of the data.' % int(float(i) / len(qpoint_lists) * 100)

    print 'percentage of point collision: ' + str(float(pcol)/ps)
    print 'number of samples: ' +str(len(f['data_set/data_set']))
    print 'dimensionality of the samples: ' +str(len(f['data_set/data_set'][0]))
    print 'number of labels: ' +str(len(f['labels/real_labels']))
    print 'number of annotations: ' +str(len(f['annotations/annotations']))

    f.close()


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

def generate_train_val_test_set():
    ############################################### SCALING DATA AND GENERATING TRAINING AND VALIDATION SET ######################################

    file = h5.File("usarray_data_unscaled_real.hdf5", "r")

    samples = scale(file['data_set/data_set'][...])
    labels = file['labels/real_labels'][...]
    annotations = file['annotations/annotations'][...]


    train_set = []
    train_labels = []
    train_annotations = []

    val_set = []
    val_labels = []
    val_annotations = []

    test_set = []
    test_labels = []
    test_annotations = []

    for i in np.arange(len(samples)):
        rand =  np.random.random_sample()
        if rand <= 0.6:
            train_set.append(samples[i])
            train_labels.append(labels[i])
            train_annotations.append(annotations[i])
        elif rand > 0.6 and rand <= 0.8 :
            val_set.append(samples[i])
            val_labels.append(labels[i])
            val_annotations.append(annotations[i])
        else:
            test_set.append(samples[i])
            test_labels.append(labels[i])
            test_annotations.append(annotations[i])


    print 'training set has %i samples.' %len(train_set)
    print 'training set has %i labels.' %len(train_labels)
    print 'training set has %i annotations.' %len(train_annotations)
    print 'validation set has %i samples. ' %len(val_set)
    print 'validation set has %i labels.' %len(val_labels)
    print 'validation set has %i annotations.' %len(val_annotations)
    print 'test set has %i samples. ' %len(test_set)
    print 'test set has %i labels.' %len(test_labels)
    print 'test set has %i annotations.' %len(test_annotations)

    file.close()

    ########################################################################################################################

    train_len = len(train_set) - len(train_set)%10000
    val_len = len(val_set) - len(val_set)%10000
    test_len = len(test_set) - len(test_set)%10000
    dim = len(train_set[0])

    print 'length of training set after pruning: %i' %train_len
    print 'length of validation set after pruning: %i' %val_len
    print 'length of test set after pruning: %i' %test_len

    f = h5.File("usarray_data_scaled_train_val_test_real.hdf5", "w")
    f.create_dataset('trainig_set/train_set', (train_len,dim), dtype='f')
    f.create_dataset('validation_set/val_set', (val_len,dim), dtype='f')
    f.create_dataset('test_set/test_set', (test_len,dim), dtype='f')
    f.create_dataset('trainig_labels/real_train_labels', (train_len,), dtype='i')
    f.create_dataset('validation_labels/real_val_labels', (val_len,), dtype='i')
    f.create_dataset('test_labels/real_test_labels', (test_len,), dtype='i')
    dt = h5.special_dtype(vlen=unicode)
    f.create_dataset('training_annotations/train_annotations', (train_len,), dtype=dt)
    f.create_dataset('validation_annotations/val_annotations', (val_len,), dtype=dt)
    f.create_dataset('test_annotations/test_annotations', (test_len,), dtype=dt)


    f['trainig_set/train_set'][...] = train_set[:train_len]
    f['validation_set/val_set'][...] = val_set[:val_len]
    f['test_set/test_set'][...] = test_set[:test_len]
    print 'created data sets.'
    f['trainig_labels/real_train_labels'][...] = train_labels[:train_len]
    f['validation_labels/real_val_labels'][...] = val_labels[:val_len]
    f['test_labels/real_test_labels'][...] = test_labels[:test_len]
    print 'created labels.'
    f['training_annotations/train_annotations'][...] = train_annotations[:train_len]
    f['validation_annotations/val_annotations'][...] = val_annotations[:val_len]
    f['test_annotations/test_annotations'][...] = test_annotations[:test_len]
    print 'created annotations.'

    f.close()

if __name__ == '__main__':
    with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            generate_real_dataset_binning()
            #generate_train_val_test_set()

