#!/usr/bin/env python
import rospy
import numpy as np
import cPickle as cp

from utils import determine_label
from sensor_msgs.msg import PointCloud2
from point_cloud import read_points
from breze.learn.cnn import Cnn
from breze.learn.mlp import Mlp
from markov_chain import Markov_Chain


class Predictor:

    # initialize the object
    def __init__(self):
        with open('config.txt', 'r') as config_f:
            for line in config_f:
                if not line.find('mode='):
                    self.mode = line.replace('mode=', '').replace('\n', '')
                if not line.find('robust='):
                    self.robust = line.replace('robust=', '').replace('\n', '')
        print 'mode=%s\nrobustness=%s' %(self.mode, self.robust)

        if self.robust == 'majority':
            self.pred_count = 0
            self.predictions = np.zeros((13,))
        if self.robust == 'markov':
            self.markov = Markov_Chain()
            self.last_state = 0
            self.current_state = 0

        self.sample_count = 0
        self.sample = []

        if self.mode == 'cnn':
            self.bin_cm = 10
            self.max_x_cm = 440
            self.min_x_cm = 70
            self.max_y_cm = 250
            self.max_z_cm = 200
            self.nr_z_intervals = 2
            self.x_range = (self.max_x_cm - self.min_x_cm)/self.bin_cm
            self.y_range = self.max_y_cm*2/self.bin_cm
            self.z_range = self.nr_z_intervals
            self.input_size = 3700
            self.output_size = 13
            self.n_channels = 2
            self.im_width = self.y_range
            self.im_height = self.x_range

            print 'initializing cnn model.'
            self.model = Cnn(self.input_size, [16, 32], [200, 200], self.output_size, ['tanh', 'tanh'], ['tanh', 'tanh'],
                        'softmax', 'cat_ce', image_height=self.im_height, image_width=self.im_width,
                        n_image_channel=self.n_channels, pool_size=[2, 2], filter_shapes=[[5, 5], [5, 5]], batch_size=1)
            self.model.parameters.data[...] = cp.load(open('./best_cnn_pars.pkl', 'rb'))

        if self.mode == 'crafted':
            self.input_size = 156
            self.output_size = 13
            self.means = cp.load(open('means_crafted.pkl', 'rb'))
            self.stds = cp.load(open('stds_crafted.pkl', 'rb'))

            print 'initializing crafted features model.'
            self.model = Mlp(self.input_size, [1000, 1000], self.output_size, ['tanh', 'tanh'], 'softmax', 'cat_ce',
                             batch_size=1)
            self.model.parameters.data[...] = cp.load(open('./best_crafted_pars.pkl', 'rb'))

        # this is just a trick to make the internal C-functions compile before the first real sample arrives
        compile_sample = np.random.random((1,self.input_size))
        self.model.predict(compile_sample)

        print 'starting to listen to topic.'
        self.listener()

    # build the full samples from the arriving point clouds
    def build_samples(self, sample_part):
        for point in read_points(sample_part):
            self.sample.append(point)

        self.sample_count += 1

        if self.sample_count == 6:
            if self.mode == 'cnn':
                self.cnn_predict()
            if self.mode == 'crafted':
                self.crafted_predict()
            self.sample = []
            self.sample_count = 0

    # start listening to the point cloud topic
    def listener(self):
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber("/USArray_pc", PointCloud2, self.build_samples)
        rospy.spin()

    # let the model predict the output
    def cnn_predict(self):
        grid = np.zeros((self.z_range, self.x_range, self.y_range))

        for point in self.sample:
            if point[0]*100 < self.min_x_cm or point[0]*100 > self.max_x_cm-1 or point[1]*100 > self.max_y_cm-1 or point[1]*100 < -self.max_y_cm:
                continue

            x = (int(point[0]*100) - self.min_x_cm) / self.bin_cm
            y = (int(point[1]*100) + self.max_y_cm) / self.bin_cm
            z = int(point[2]*100) > (self.max_z_cm / self.nr_z_intervals)
            pow = point[4]

            if grid[z][x][y] != 0:
                if grid[z][x][y] < pow:
                    grid[z][x][y] = pow
            else:
                grid[z][x][y] = pow

        grid = np.reshape(grid,(1,-1))

        self.output_prediction(self.model.predict(grid))


    # let the model predict the output
    def crafted_predict(self):
        vec = np.zeros((156,), dtype=np.float32)
        area_points = [[] for _ in np.arange(12)]
        area_counts = np.zeros(12)
        area_x_means = np.zeros(12)
        area_y_means = np.zeros(12)
        area_z_means = np.zeros(12)
        area_highest = np.zeros(12)
        area_highest_pow = np.zeros(12)
        area_pow_means = np.zeros(12)
        area_x_vars = np.zeros(12)
        area_y_vars = np.zeros(12)
        area_z_vars = np.zeros(12)
        area_xy_covars = np.zeros(12)
        area_xz_covars = np.zeros(12)
        area_yz_covars = np.zeros(12)
        bad = False

        for qpoint in self.sample:
            # need to substract -1 since the function returns the value starting with 1
            label = determine_label((float(qpoint[0]), float(qpoint[1]), float(qpoint[2])))-1
            area_points[label].append(qpoint)
            area_counts[label] += 1
            if float(qpoint[2]) > area_highest[label]:
                area_highest[label] = float(qpoint[2])
            if float(qpoint[4]) > area_highest_pow[label]:
                area_highest_pow[label] = float(qpoint[4])

        for area in np.arange(12):
            for point in area_points[area]:
                area_x_means[area] += float(point[0])
                area_y_means[area] += float(point[1])
                area_z_means[area] += float(point[2])
                area_pow_means[area] += float(point[4])
            if area_counts[area] > 0:
                area_x_means[area] /= area_counts[area]
                area_y_means[area] /= area_counts[area]
                area_z_means[area] /= area_counts[area]
                area_pow_means[area] /= area_pow_means[area]

            for point in area_points[area]:
                area_x_vars[area] += (float(point[0]) - area_x_means[area])**2
                area_y_vars[area] += (float(point[1]) - area_y_means[area])**2
                area_z_vars[area] += (float(point[2]) - area_z_means[area])**2
            # if there is only one point, we assume the uncorrected estimator and implicitly divide by one
            if area_counts[area] > 1:
                area_x_vars[area] *= 1/(area_counts[area]-1)
                area_y_vars[area] *= 1/(area_counts[area]-1)
                area_z_vars[area] *= 1/(area_counts[area]-1)

            for point in area_points[area]:
                area_xy_covars[area] += (float(point[0]) - area_x_means[area])*(float(point[1]) - area_y_means[area])
                area_xz_covars[area] += (float(point[0]) - area_x_means[area])*(float(point[2]) - area_z_means[area])
                area_yz_covars[area] += (float(point[1]) - area_y_means[area])*(float(point[2]) - area_z_means[area])
            # if there is only one point, we assume the uncorrected estimator and implicitly divide by one
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
            vec[area*11+11] = area_highest_pow[area]
            vec[area*11+12] = area_pow_means[area]

        vec = np.reshape(vec, (1, 156))
        vec -= self.means
        vec /= self.stds

        self.output_prediction(self.model.predict(vec))

    # create the output
    def output_prediction(self, probabilites):
        if self.robust == 'majority':
            prediction = np.argmax(probabilites)
            # majority vote among the last three predictions
            self.predictions[prediction] += 1
            self.pred_count += 1
            if self.pred_count == 3:
                print 'majority prediction: %d' %np.argmax(self.predictions)
                self.pred_count = 0
                self.predictions = np.zeros((13,))
        if self.robust == 'markov':
            markov_probs = self.markov.transition_table[self.last_state]
            probabilites *= markov_probs
            probabilites /= np.sum(probabilites)
            prediction = np.argmax(probabilites)
            print 'markov prediction: %d' %prediction
            self.last_state = prediction
        if self.robust == 'off':
            prediction = np.argmax(probabilites)
            print 'fast prediction: %d' %prediction

        
if __name__ == '__main__':
    predictor = Predictor()
