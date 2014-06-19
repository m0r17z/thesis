
import numpy as np
import h5py as h5
from utils import determine_label
from generate_datasets import generate_train_val_test_set

def generate_real_dataset_crafted():
    ################################################ LOADING AND CLEANING THE DATA #########################################
    samples = open('/nthome/maugust/thesis/samples.txt')
    labels = open('/nthome/maugust/thesis/labels.txt')
    annotations = open('/nthome/maugust/thesis/annotations.txt')

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

    ################################################## COMPUTE THE FEATURES ###########################################
    f = h5.File("./crafted_real.hdf5", "w")
    f.create_dataset('data_set/data_set', (len(qpoint_lists),132), dtype='f')
    f.create_dataset('labels/real_labels', (len(real_labels),), dtype='i')
    dt = h5.special_dtype(vlen=unicode)
    f.create_dataset('annotations/annotations', (len(annotation_list),), dtype=dt)

    last_per = -1

    for ind, qpoint_list in enumerate(qpoint_lists):
        vec = np.zeros((132,))
        area_points = [[] for _ in np.arange(12)]
        area_counts = np.zeros(12)
        area_x_means = np.zeros(12)
        area_y_means = np.zeros(12)
        area_z_means = np.zeros(12)
        area_highest = np.zeros(12)
        area_x_vars = np.zeros(12)
        area_y_vars = np.zeros(12)
        area_z_vars = np.zeros(12)
        area_xy_covars = np.zeros(12)
        area_xz_covars = np.zeros(12)
        area_yz_covars = np.zeros(12)

        for qpoint in qpoint_list:
            # need to substract -1 since the function returns the value starting with 1
            label = determine_label((float(qpoint[0]), float(qpoint[1]), float(qpoint[2])))-1
            area_points[label].append(qpoint)
            area_counts[label] += 1
            if float(qpoint[2]) > area_highest[label]:
                area_highest[label] = float(qpoint[2])

        for area in np.arange(12):
            for point in area_points[area]:
                area_x_means[area] += float(point[0])
                area_y_means[area] += float(point[1])
                area_z_means[area] += float(point[2])
            if area_counts[area] != 0:
                area_x_means[area] /= area_counts[area]
                area_y_means[area] /= area_counts[area]
                area_z_means[area] /= area_counts[area]

            for point in area_points[area]:
                area_x_vars[area] += (float(point[0]) - area_x_means[area])**2
                area_y_vars[area] += (float(point[1]) - area_y_means[area])**2
                area_z_vars[area] += (float(point[2]) - area_z_means[area])**2
            # if there is only one point, we assume the uncorrected estimator and divide by one
            if area_counts[area] > 1:
                area_x_vars[area] *= 1/(area_counts[area]-1)
                area_y_vars[area] *= 1/(area_counts[area]-1)
                area_z_vars[area] *= 1/(area_counts[area]-1)

            for point in area_points[area]:
                area_xy_covars[area] += (float(point[0]) - area_x_means[area])*(float(point[1]) - area_y_means[area])
                area_xz_covars[area] += (float(point[0]) - area_x_means[area])*(float(point[2]) - area_z_means[area])
                area_yz_covars[area] += (float(point[1]) - area_y_means[area])*(float(point[2]) - area_z_means[area])
            # if there is only one point, we assume the uncorrected estimator and divide by one
            if area_counts[area] > 1:
                area_xy_covars[area] *= 1/(area_counts[area]-1)
                area_xz_covars[area] *= 1/(area_counts[area]-1)
                area_yz_covars[area] *= 1/(area_counts[area]-1)

        for area in np.arange(12):
            vec[area*11] = area_counts[area]
            vec[area*11+1] = area_x_means[area]
            vec[area*11+2] = area_y_means[area]
            vec[area*11+3] = area_z_means[area]
            vec[area*11+4] = area_x_vars[area]
            vec[area*11+5] = area_y_vars[area]
            vec[area*11+6] = area_z_vars[area]
            vec[area*11+7] = area_xy_covars[area]
            vec[area*11+8] = area_xz_covars[area]
            vec[area*11+9] = area_yz_covars[area]
            vec[area*11+10] = area_highest[area]


        # unroll the grid into a vector?!
        f['data_set/data_set'][ind] = vec.flatten()
        f['labels/real_labels'][ind] = real_labels[ind]
        f['annotations/annotations'][ind] = annotation_list[ind]
        curr_percent = int(float(ind) / len(qpoint_lists) * 100)
        if last_per != curr_percent:
            last_per = curr_percent
            print 'have now looked at %i%% of the data.' % int(float(ind) / len(qpoint_lists) * 100)

    print 'number of samples: ' +str(len(f['data_set/data_set']))
    print 'dimensionality of the samples: ' +str(len(f['data_set/data_set'][0]))
    print 'number of labels: ' +str(len(f['labels/real_labels']))
    print 'number of annotations: ' +str(len(f['annotations/annotations']))

    f.close()

    generate_train_val_test_set("./crafted_real.hdf5", "train_val_test_crafted_real.hdf5")


if __name__ == '__main__':
    generate_real_dataset_crafted()