import numpy as np
import matplotlib.pyplot as plt


def visualize_output(output):
    if not output.shape == (3,4):
        print 'ERROR: output has unexpected shape %s' %str(output.shape)
        return -1
    fig, ax = plt.subplots()

    ax.set_title('Probabilities for the Areas')

    ax.imshow(output, cmap=plt.cm.gray, interpolation='nearest')
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    return fig, ax

def compare_prediciton_target(prediction, target):
    if not prediction.shape == (3,4):
        print 'ERROR: output has unexpected shape %s' %str(prediction.shape)
        return -1

    if not target.shape == (3,4):
        print 'ERROR: output has unexpected shape %s' %str(target.shape)
        return -1

    fig, ax = plt.subplots(1,2)

    ax[0].set_title('Predicted Probabilities')

    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[0].imshow(prediction, cmap=plt.cm.gray, interpolation='nearest')

    ax[1].set_title('Hard Label')

    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    ax[1].imshow(target, cmap=plt.cm.gray, interpolation='nearest')

    plt.show()

if __name__ == '__main__':
    compare_prediciton_target(np.random.uniform(size=(3, 4)),np.zeros((3,4)))