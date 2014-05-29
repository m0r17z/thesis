
import h5py as h5
import numpy as np



def generate_train_val_test_set(raw_data, final_data):
    ############################################### SCALING DATA AND GENERATING TRAINING AND VALIDATION SET ######################################

    file = h5.File(raw_data, "r")

    samples = file['data_set/data_set'][...]
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

    f = h5.File(final_data, "w")
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

