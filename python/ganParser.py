import os, sys, math, matplotlib, json, random
import matplotlib.pyplot as plt

def drawAllGAN(folder):
    """
    Loop through all files of a given folder
    and call drawFromGAN with text files that
    contain json with at least "sample":[[[x1,y1],[x2,y2],...,[x_n,y_n]]]

    Parameters
    ----------
    folder : string
        The folder where all text files are located.
    """
    for file in os.listdir(folder):
        if ".jpg" in file or ".js" in file or ".ts" in file or "epoch" in file or "html" in file : continue
        with open(folder + os.sep + file) as json_file:
            data = json.load(json_file)
        drawFromGAN(data, folder + os.sep + file, xScale=1536, yScale=864)

def drawFromGAN(data, name, xScale=1, yScale=1, show=False):
    """
    Redraw a sample output from an GAN generator
    with matplotlib, the data object is a dict
    with at least "sample":[[[x1,y1],[x2,y2],...,[x_n,y_n]]].

    Parameters
    ----------
    data : dict
        A dict with a key 'sample' that contains the list of points
        of the trajectory.
    name : string
        The name of the jpg file of the image.
    xScale : int
        The x scale factor to divide all x values.
    yScale : int
        The y scale factor to divide all y values.
    show : boolean
        Default to false, if true then the image is shown.
    """
    samples = data['sample'][0]
    x_values, y_values = [], []
    for point in samples:
        point[0] *= xScale
        point[1] *= yScale
        x_values.append(point[0])
        y_values.append(point[1])
        plt.plot(point[0], point[1], marker='o', color='red')

    plt.plot(x_values, y_values, color='blue')

    plt.savefig(name + '.jpg')
    if show: plt.show()
    plt.close()

def drawLossAcc(folder, show=False):
    """
    Redraw the loss and accuracy of a GAN training.
    At each epoch, we have a text file that represents
    a json :
    {
      "discriminatorLoss":[loss,accuracy],
      "generatorLoss":[loss,accuracy],
      "sample":[[[x1,y1],[x2,y2],...,[x_n,y_n]]]
    }

    This function zip all loss and accuracy values
    to show them in two matplotlib plots, one for
    loss and the other for accuracy.

    Parameters
    ----------
    folder : string
        The folder where all text files are located.
    show : boolean
        Default to false, if true then the image is shown.
    """
    discriminatorLoss = []
    generatorLoss = []
    discriminatorAcc = []
    generatorAcc = []
    i = 0
    file = f"{folder}/{folder.split('/')[1]}_{i}.txt"
    while os.path.isfile(file):
        with open(file) as json_file:
            data = json.load(json_file)
        discriminatorLoss.append(data["discriminatorLoss"][0])
        generatorLoss.append(data["generatorLoss"][0])
        discriminatorAcc.append(data["discriminatorLoss"][1])
        generatorAcc.append(data["generatorLoss"][1])
        i += 1
        file = f"{folder}/{folder.split('/')[1]}_{i}.txt"

    plt.plot(range(i), discriminatorLoss, color='blue', label='Discriminator loss')
    plt.plot(range(i), generatorLoss, color='red', label='Generator loss')

    plt.legend()

    plt.savefig(folder + os.sep + 'loss.jpg')
    if show: plt.show()
    plt.close()


    plt.plot(range(i), discriminatorAcc, color='blue', label='Discriminator accuracy')
    plt.plot(range(i), generatorAcc, color='red', label='Generator accuracy')

    plt.legend()

    plt.savefig(folder + os.sep + 'l_accuracy.jpg')
    if show: plt.show()
    plt.close()

def parseSessionGAN():
    """
    List all GAN folder and parse samples from
    their files if the folder doesn't contain "non"
    in their names.
    """
    for folder in os.listdir("GAN/"):
        folder = "GAN/" + folder
        if not os.path.isdir(folder): continue
        if "non" in folder: continue
        for file in os.listdir(folder):
            if ".jpg" in file or ".js" in file or ".ts" in file or "epoch" in file or "html" in file : continue
            with open(f"{folder}/{file}") as json_file:
                data = json.load(json_file)
            writeSession(file, data['sample'][0])

def writeSession(file, samples, xScale=1536, yScale=864):
    """
    Writes in circles_forbidden_gan/<file> the
    trajectory of the given GAN samples.
    To simulate time, we set a random time offset
    between each movement (from 5 to 23ms).

    Parameters
    ----------
    file : string
        The name of the file to write the trajectory.
    samples : list
        A list of samples where each element is a tuple
        of coordinates (x,y).
    xScale : int
        The x scale factor to multiply all x values.
    yScale : int
        The y scale factor to multiply all y values.
    """
    with open(f"circles_forbidden_gan/{file}", "w") as f:
        f.write(f"resolution:{xScale},{yScale}\n")
        f.write(f"0,Pressed,{samples[0][0]*xScale},{samples[0][1]*yScale}\n")
        time = random.uniform(5, 23)
        for point in samples:
            f.write(f"{time},Move,{point[0]*xScale},{point[1]*yScale}\n")
            time += random.uniform(5, 23)
        f.write(f"{time},Released,{samples[-1][0]*xScale},{samples[-1][1]*yScale}")


if __name__ == "__main__":
    parseSessionGAN()
    exit()

    folder = 'GAN/GAN36'
    matplotlib.use('Agg')

    drawLossAcc(folder)
    drawAllGAN(folder)
