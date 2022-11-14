import os, sys, math, matplotlib
import matplotlib.pyplot as plt

def readSession(file, normalize=False) -> list:
    """
    Read and parse a session file, containing
    trajectories where each line contains
    'timestamp,type,x,y' and the first line
    describe the screen resolution to normalize
    all x and y. For example, a valid format is :

    resolution:1536,864
    9131.1,Pressed,717,361
    9134.8,Move,717,361
    9151.8,Move,717,361
    ...
    10402.3,Move,722,360
    10419.1,Move,718,358
    10425.8,Released,717,360

    The parser gets all lines and returns a list
    of all coordinates where an object of this list
    is a dict with keys x,y,type,time,speed.

    Parameters
    ----------
    file : string
        Path to the sample file.
    normalize : boolean
        Default to false, if true and the first line
        is "resolution:x;y" then we ignore it.

    Returns
    -------
    list
        A list of dict containing the input file.
    """
    parsedSession = []
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

            parsedLine = {
                'time':float(line[0]),
                'x':float(line[2]) / screenX,
                'y':float(line[3]) / screenY,
                'type':line[1]
            }
            if "Move" in parsedLine['type']:
                if prev_line == None:
                    parsedLine['speed'] = 0
                else:
                    dx = parsedLine['x'] - prev_line['x']
                    dy = parsedLine['y'] - prev_line['y']
                    dist = math.sqrt(dx*dx + dy*dy)
                    time = parsedLine['time'] - prev_line['time']
                    parsedLine['speed'] = 0 if time == 0 else dist/time

                prev_line = parsedLine

            parsedSession.append(parsedLine)

    return parsedSession

def reDraw(parsedSession, sessName, show=False):
    """
    Get a parsed session loaded from readSession()
    and draw the trajectory with matplotlib.

    Parameters
    ----------
    parsedSession : list
        The list of dict returned by readSession().
    sessName : string
        The name of the jpg file of the image.
    show : boolean
        Default to false, if true then the image is shown.
    """
    draw = False
    x_values, y_values = [], []
    for line in parsedSession:
        if "Move" in line['type'] and draw:
            x_values.append(line['x'])
            y_values.append(line['y'])
            plt.plot(line['x'], line['y'], marker='o', color='red')
        elif "Released" in line['type']:
            plt.plot(x_values, y_values, color='blue')
            #plt.plot(line['x'], line['y'], marker='o', color='green')
            x_values.clear()
            y_values.clear()
            draw = False
        elif "Pressed" in line['type']:
            #plt.plot(line['x'], line['y'], marker='o', color='red')
            draw = True

    plt.savefig(sessName + '.jpg')
    if show: plt.show()
    #plt.figure().clear()
    plt.close()

def velocity(parsedSession, sessName, show=False):
    """
    Get a parsed session loaded from readSession()
    and draw the velocity function with matplotlib.

    Parameters
    ----------
    parsedSession : list
        The list of dict returned by readSession().
    sessName : string
        The name of the jpg file of the image.
    show : boolean
        Default to false, if true then the image is shown.
    """
    time, speed = [], []
    for line in parsedSession:
        if not "Move" in line['type']: continue
        time.append(line['time'])
        speed.append(line['speed'])

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
