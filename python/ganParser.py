import os, sys, math, matplotlib, json, random
import matplotlib.pyplot as plt

def drawAllGAN(folder):
    for file in os.listdir(folder):
        if ".jpg" in file or ".js" in file or ".ts" in file or "epoch" in file or "html" in file : continue
        with open(folder + os.sep + file) as json_file:
            data = json.load(json_file)
        drawFromGAN(data, folder + os.sep + file, xScale=1536, yScale=864)

def drawFromGAN(data, name, xScale=1, yScale=1, show=False):
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
    #plt.figure().clear()
    plt.close()

def drawLossAcc(folder, show=False):
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
    with open(f"circles_forbidden_gan/{file}", "w") as f:
        f.write(f"resolution:{xScale},{yScale}\n")
        f.write(f"0,0,Button,Pressed,{samples[0][0]*xScale},{samples[0][1]*yScale}\n")
        time = random.uniform(5, 23)
        for point in samples:
            f.write(f"{time},{time},NoButton,Move,{point[0]*xScale},{point[1]*yScale}\n")
            time += random.uniform(5, 23)
        f.write(f"{time},{time},Button,Released,{samples[-1][0]*xScale},{samples[-1][1]*yScale}")


if __name__ == "__main__":
    parseSessionGAN()
    exit()

    folder = 'GAN/GAN36'
    matplotlib.use('Agg')

    drawLossAcc(folder)
    drawAllGAN(folder)