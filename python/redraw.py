import os, sys, math, matplotlib
import matplotlib.pyplot as plt

def readSession(file, normalize=False):
    operations = []
    with open(file, 'r') as f:
        prev_line = None
        for line in f:
            if "resolution" in line:
                if normalize:
                    screenX, screenY = list(map(int, line.split(":")[1].split(",")))
                else:
                    screenX, screenY = 1, 1
                continue
            line = line.strip().split(",")
            line[0] = float(line[0])
            line[4] = float(line[4]) / screenX
            line[5] = float(line[5]) / screenY

            if "Move" in line[3]:
                if prev_line == None:
                    line.append(0)
                else:
                    dx = line[4] - prev_line[4]
                    dy = line[5] - prev_line[5]
                    dist = math.sqrt(dx*dx + dy*dy)
                    time = line[0] - prev_line[0]
                    line.append(0 if time == 0 else dist/time)
                prev_line = line

            operations.append(line)

    return operations

def reDraw(parsedSession, sessName, show=False):
    draw = False
    x_values, y_values = [], []
    for action in parsedSession:
        if "Move" in action[3] and draw:
            x_values.append(action[4])
            y_values.append(action[5])
            plt.plot(action[4], action[5], marker='o', color='red')
        elif "Released" in action[3]:
            plt.plot(x_values, y_values, color='blue')
            #plt.plot(action[4], action[5], marker='o', color='green')
            x_values.clear()
            y_values.clear()
            draw = False
        elif "Pressed" in action[3]:
            #plt.plot(action[4], action[5], marker='o', color='red')
            draw = True

    plt.savefig(sessName + '.jpg')
    if show: plt.show()
    #plt.figure().clear()
    plt.close()

def velocity(parsedSession, sessName, show=False):
    time, speed = [], []
    for action in parsedSession:
        if not "Move" in action[3]: continue
        time.append(action[0])
        speed.append(action[6])

    plt.plot(time, speed, color='blue')
    plt.savefig(sessName + '_velocity.jpg')
    if show: plt.show()
    #plt.figure().clear()
    plt.close()


if __name__ == "__main__":
    normalize = len(sys.argv) >= 2 and sys.argv[1].lower() == "true"
    matplotlib.use('Agg')
    for folder in ["circles_bot_naturalmousemotion"]:
        print(">>", folder, "...")
        for file in os.listdir(folder):
            if not ".txt" in file or ".jpg" in file: continue
            name = folder + os.sep + file
            sess = readSession(name, normalize=normalize)
            reDraw(sess, name)
            #velocity(sess, name)
