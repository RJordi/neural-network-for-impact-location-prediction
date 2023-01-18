import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def plot_heatmap(prediction, real_label, labListA, interp_prediction):

    x_coord = [x for x in range(175, 330, 5)]
    y_coord = [y for y in range(175, 330, 5)]
    probabilities = np.zeros((len(x_coord), len(y_coord)))

    for i in range(len(prediction)):
        probabilities[x_coord.index(labListA[1][i]+250)][x_coord.index(labListA[0][i]+250)] = prediction[i]


    fig, ax = plt.subplots()
    im = ax.imshow(probabilities)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Probability of impact', rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(x_coord)))
    ax.set_yticks(np.arange(len(y_coord)))

    ax.set_xticklabels(x_coord)
    ax.set_yticklabels(y_coord)


    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
         rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    #for i in range(len(x_coord)):
    #    for j in range(len(y_coord)):
    #        if probabilities[i,j] > 0:
    #            text = ax.text(x_coord.index(real_label[0]), x_coord.index(real_label[1]), round(probabilities[i, j], 2),
    #                           ha="center", va="center", color="w")

    # Plot interpolated predicted coordinate and real coordinate
    ax.text(x_coord.index(real_label[0]), x_coord.index(real_label[1]), 'X', ha="center", va="center", color='w')
    ax.text(x_coord.index(int(np.around(interp_prediction[0]/5,decimals=0)*5)), x_coord.index(int(np.around(interp_prediction[1]/5,decimals=0)*5)), 'o', ha="center", va="center", color='r')
    #plt.plot(x_coord.index(real_label[0]), x_coord.index(real_label[1]), 'ro')
    real_coord = plt.scatter([], [], color='w', marker='x', label='Real impact position')
    interp_coord = plt.scatter([], [], edgecolors='r', marker='.', facecolors='none', label='Predicted impact position', s=100)
    plt.legend(handles=[real_coord, interp_coord])

    ax.set_title("Probabilities of impact location")
    fig.tight_layout()
    plt.show()
