import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 18})


def plot_original_data():
    df = pd.read_csv("data.csv")
    plt.scatter(df['Spendings'], df['Sales'], color='#1f77b4',marker = 'o')
    plt.xlabel("Spendings, M$")
    plt.ylabel("Sales, Units")
    plt.title("Sales vs radio ad spendings")
    #plt.show()
    axes = plt.gca()
    axes.set_xlim([0,50])
    axes.set_ylim([0,35])
    plt.tight_layout()
    fig1 = plt.gcf()
    nfig = fig2data(fig1)
    seq.append(nfig)
##    fig1.subplots_adjust(top = 0.98, bottom = 0.1, right = 0.98, left = 0.08, hspace = 0, wspace = 0)
##    fig1.savefig('../../Illustrations/gradient_descent-1.eps', format='eps', dpi=1000, bbox_inches = 'tight', pad_inches = 0)
##    fig1.savefig('../../Illustrations/gradient_descent-1.pdf', format='pdf', dpi=1000, bbox_inches = 'tight', pad_inches = 0)
##    fig1.savefig('../../Illustrations/gradient_descent-1.png', dpi=1000, bbox_inches = 'tight', pad_inches = 0)

def update_w_and_b(spendings,sales,w,b,alpha):
    dr_dw = 0.0
    dr_db = 0.0
    N = len(spendings)
    for i in range(N):
        dr_dw += -2*spendings[i]*(sales[i] - (w*spendings[i] + b))
        dr_db += -2*(sales[i] - (w*spendings[i] + b))
    # update w and b
    w = w - (dr_dw/float(N))*alpha
    b = b - (dr_db/float(N))*alpha
    return w,b

def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGB values
    """
    # draw the renderer
    fig.canvas.draw()
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.array(fig.canvas.renderer._renderer)
    return buf

seq = []
plot_original_data()
def train(spendings,sales,w,b,alpha,epochs):
    image_counter = 2;
    for e in range(epochs):
        w, b = update_w_and_b(spendings,sales,w,b,alpha)
        # log the progress
        if (e==0) or (e<3000 and e%400==0) or (e%3000==0):
            print("epoch: ", str(e), "loss: "+str(loss(spendings,sales,w,b)))
            print("w, b: ",w,b)
            plt.figure(image_counter)
            plt.xlabel("Spendings, M$")
            plt.ylabel("Sales, Units")
            axes = plt.gca()
            axes.set_xlim([0,50])
            axes.set_ylim([0,35])
            plt.scatter(spendings,sales,color='#1f77b4',marker='o' )
            X_plot = np.linspace(0,50,50)
            plt.plot(X_plot,X_plot*w + b)
            heading = 'epoch = '+str(e)+' loss = '+str(round(loss(spendings,sales,w,b)))
            plt.title(heading)
            #plt.show()
            plt.tight_layout()
            fig1 = plt.gcf()
            nfig = fig2data(fig1)
            seq.append(nfig)
##            fig1.subplots_adjust(top = 0.98, bottom = 0.1, right = 0.98, left = 0.08, hspace = 0, wspace = 0)
##            fig1.savefig('../../Illustrations/gradient_descent-' + str(image_counter) + '.eps', format='eps', dpi=1000, bbox_inches = 'tight', pad_inches = 0)
##            fig1.savefig('../../Illustrations/gradient_descent-' + str(image_counter) + '.pdf', format='pdf', dpi=1000, bbox_inches = 'tight', pad_inches = 0)
##            fig1.savefig('../../Illustrations/gradient_descent-' + str(image_counter) + '.png', dpi=1000, bbox_inches = 'tight', pad_inches = 0)
            image_counter += 1
    return w,b

def loss(spendings,sales,w,b):
    N = len(spendings)
    total_error = 0.0
    for i in range(N):
        total_error += (sales[i] - (w*spendings[i] + b))**2
    return total_error/N

df = pd.read_csv("data.csv")
x = df['Spendings']
y = df['Sales']
w,b = train(x,y,0.0,0.0,0.001,16000)

def predict(x,w,b):
    return w*x + b
x_new = 23.0
y_new = predict(x_new, w, b)
print(y_new)

import os
## Get the directory address of current python file
curr_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_dir) ## Set the current directory as working directory

## The package used to create gif files
import numpngw
numpngw.write_apng('gradient_descent.png',seq,delay = 750)
