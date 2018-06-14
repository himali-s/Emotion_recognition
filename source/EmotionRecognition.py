#!/usr/bin/env python

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import json
import subprocess
from sklearn import datasets
import tkinter.filedialog                       

import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

print(__doc__)

faces = datasets.fetch_olivetti_faces()

class Trainer:
    def __init__(self):
        self.results = {}
        self.imgs = faces.images
        self.index = 0

    def reset(self):
        self.results = {}
        self.imgs = faces.images
        self.index = 0

    def increment_face(self):
        if self.index + 1 >= len(self.imgs):
            return self.index
        else:
            while str(self.index) in self.results:
                # print self.index
                self.index += 1
            return self.index

    def record_result(self, smile=True):
        print("Image", self.index + 1, ":", "Happy" if smile is True else "Sad")
        self.results[str(self.index)] = smile

def run_once(m):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return m(*args, **kwargs)
    wrapper.has_run = False
    return wrapper

def smileCallback():
    trainer.record_result(smile=True)
    trainer.increment_face()
    displayFace(trainer.imgs[trainer.index])
    updateImageCount(happyCount=True, sadCount= False)


def noSmileCallback():
    trainer.record_result(smile=False)
    trainer.increment_face()
    displayFace(trainer.imgs[trainer.index])
    updateImageCount(happyCount=False, sadCount=True)


def updateImageCount(happyCount, sadCount):
    global HCount, SCount, imageCountString, countString   # Updating only when called by smileCallback/noSmileCallback
    if happyCount is True and HCount < 40:
        HCount += 1
    if sadCount is True and SCount < 40:
        SCount += 1
    if HCount == 40 or SCount == 40:
        HCount = 0
        SCount = 0
    imageCountPercentage = str(float((trainer.index + 1) * 2.5)) \
        if trainer.index+1 < 40 else "Classification DONE! 100"
    imageCountString = "Image Index: " + str(trainer.index+1) + "/40   " + "[" + imageCountPercentage + " %]"
    labelVar.set(imageCountString)           # Updating the Label (ImageCount)
    countString = "(Happy: " + str(HCount) + "   " + "Sad: " + str(SCount) + ")\n"
    countVar.set(countString)


@run_once
def displayBarGraph(isBarGraph):
    ax[1].axis(isBarGraph)
    n_groups = 1                    # Data to plot
    Happy, Sad = (sum([trainer.results[x] == True for x in trainer.results]),
               sum([trainer.results[x] == False for x in trainer.results]))
    index = np.arange(n_groups)     # Create Plot
    bar_width = 0.5
    opacity = 0.75
    ax[1].bar(index, Happy, bar_width, alpha=opacity, color='b', label='Happy')
    ax[1].bar(index + bar_width, Sad, bar_width, alpha=opacity, color='g', label='Sad')
    ax[1].set_ylim(0, max(Happy, Sad)+10)
    ax[1].set_xlabel('Expression')
    ax[1].set_ylabel('Number of Images')
    ax[1].set_title('Training Data Classification')
    ax[1].legend()


@run_once
def printAndSaveResult():
    print(trainer.results)                       # Prints the results
    with open("../results/results.xml", 'w') as output:
        json.dump(trainer.results, output)        # Saving The Result

@run_once
def loadResult():
    results = json.load(open("../results/results.xml"))
    trainer.results = results


def displayFace(face):
    ax[0].imshow(face, cmap='gray')
    isBarGraph = 'on' if trainer.index+1 == 40 else 'off'      # Switching Bar Graph ON
    if isBarGraph is 'on':
        displayBarGraph(isBarGraph)
        printAndSaveResult()
    # f.tight_layout()
    canvas.draw()


def _opencv():
    print("\n\n Please Wait. . . .")
    opencvProcess = subprocess.Popen("Train Classifier and Test Video Feed.py", close_fds=True, shell=True)

def _begin():
    trainer.reset()
    global HCount, SCount
    HCount = 0
    SCount = 0
    updateImageCount(happyCount=False, sadCount=False)
    displayFace(trainer.imgs[trainer.index])


def _quit():
    root.quit()     # stops mainloop
    root.destroy()  
                    


if __name__ == "__main__":
    matplotlib.use('TkAgg')
    root = Tk.Tk()
    root.wm_title("Emotion Recognition Using Scikit-Learn & OpenCV")

    trainer = Trainer()

    # Creating the figure to be embedded into the tkinter plot
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(faces.images[0], cmap='gray')
    ax[1].axis('off')  # Initially keeping the Bar graph OFF

    # ax tk.DrawingArea
    # Embedding the Matplotlib figure 'f' into Tkinter canvas
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.show()
    canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

    print("Keys in the Dataset: ", faces.keys())
    print("Total Images in Olivetti Dataset: 40")

    # Declaring Button & Label Instances
    smileButton = Tk.Button(master=root, text='Smiling', command=smileCallback)
    smileButton.pack(side=Tk.LEFT)

    noSmileButton = Tk.Button(master=root, text='Not Smiling', command=noSmileCallback)
    noSmileButton.pack(side=Tk.RIGHT)

    labelVar = Tk.StringVar()
    label = Tk.Label(master=root, textvariable=labelVar)
    imageCountString = "Image Index: 0/40   [0 %]"     # Initial print
    labelVar.set(imageCountString)
    label.pack(side=Tk.TOP)

    countVar = Tk.StringVar()
    HCount = 0
    SCount = 0
    countLabel = Tk.Label(master=root, textvariable=countVar)
    countString = "(Happy: 0   Sad: 0)\n"     # Initial print
    countVar.set(countString)
    countLabel.pack(side=Tk.TOP)

    #opencvButton = Tk.Button(master=root, text='Load the "Trained Classifier" & Test Output', command=_opencv)
    #opencvButton.pack(side=Tk.TOP)

    resetButton = Tk.Button(master=root, text='Reset', command=_begin)
    resetButton.pack(side=Tk.TOP)

    quitButton = Tk.Button(master=root, text='Quit Application', command=_quit)
    quitButton.pack(side=Tk.TOP)

    Tk.mainloop()                               # Starts mainloop required by Tk
