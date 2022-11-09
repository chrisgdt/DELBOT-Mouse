import time, math, random, time, json, subprocess
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import pyautogui
import pyHM
from pyclick import HumanClicker
from pynput.mouse import Button, Controller


url = "https://chrisgdt.github.io/mouse-drawing/"
#url = "http://localhost:63342/TFJS_mouse_balabit/index.html?_ijt=dnc6ho2bpbbeahi8trv35jkg6b"

def start_driver():
    options = Options()
    #options.add_argument('--headless')
    #options.add_argument('--hide-scrollbars')
    #options.add_argument('--disable-gpu')
    options.add_argument("--log-level=3")  # fatal
    #options.add_argument("--incognito")
    #options.add_experimental_option("debuggerAddress", "127.0.0.1:55217/devtools/browser/88fbb9aa-d582-40b5-b47c-bcf3a9586d3d")
    options.binary_location = r"C:\Users\chris\AppData\Local\Programs\Opera GX\91.0.4516.95\opera.exe"
    driver = webdriver.Opera(executable_path=r"C:\webdrivers\operadriver_105.exe", options=options)

    #driver = webdriver.Chrome('C:\\webdrivers\\chromedriver.exe')
    driver.get(url)

    return driver


def click_circle(driver):
    return _open_drawing_canvas(driver, "circle")

def click_sheep(driver):
    return _open_drawing_canvas(driver, "sheep")

def click_cloud(driver):
    return _open_drawing_canvas(driver, "cloud")

def click_sun(driver):
    return _open_drawing_canvas(driver, "sun")

def _open_drawing_canvas(driver, buttonId):
    canvas = driver.find_element(By.ID, f"{buttonId}Canvas")

    show_button = driver.find_element(By.ID, buttonId)
    export_button = driver.find_element(By.ID, f"exportDraw_{buttonId}Canvas")
    close_button = driver.find_element(By.ID, f"hideDraw_{buttonId}Canvas")

    show_button.click()

    return export_button, close_button


def get_circle_points(radius, step):
    x_size, y_size = pyautogui.size()
    center = pyautogui.position(x_size/2, y_size/2.2)

    direction = 2*random.randint(0,1)-1  # -1 or 1
    offset_angle = random.uniform(0,2*math.pi)  # start at random pos on circle
    ellipsis = random.uniform(0, .4)
    offset_x = random.uniform(.8, 1.2)
    offset_y = random.uniform(.8, 1.2)

    points = []

    inc = math.pi*2/step
    for i in range(step):
        radius *= random.uniform(.99, 1.01)
        angle = i * inc
        points.append((center[0]+offset_x*radius*math.cos(angle+offset_angle),
                       center[1]+offset_y*radius*math.sin(angle+offset_angle+ellipsis)*direction))

    return points


def wait_linear(duration, step, i):
    # total time : step*duration, linear steps
    time.sleep(duration)

def wait_quadratic(duration, step, i):
    # quadratric steps
    time.sleep(duration/math.sqrt(i))

def wait_gaussian(duration, step, i):
    # at i=0 : duration, at i=step/2 : 0, at i=step : duration
    begin = duration + math.pow(math.e, -duration*step/4)
    center = -math.log(begin)
    val = begin - math.pow(math.e, -center - duration / step * math.pow(i-step/2, 2))
    time.sleep(max(0, val))


def circle_automated(driver, move_type=5):
    export_button, close_button = click_circle(driver)
    get_points_function = get_circle_points

    if move_type == 1:
        for radius in [100, 200, 250, 300, 330]:
            for step in [15, 30, 50, 80, 100, 150, 250]:
                for duration in [.15, .1, .05, .01, .005, .001]:
                    for wait in range(3):
                        points = get_points_function(radius, step)
                        pyautogui.moveTo(*points[0])
                        pyHM.mouse.down()

                        for point in points:
                            # Tweens : ease(In|Out|InOut)(Quad|Cubic|Quart|Quint|Sine|Expo|Circ|Elastic|Back|Bounce)
                            # Default tween : linear
                            if wait == 0:
                                tween = pyautogui.linear
                            elif wait == 1:
                                tween = random.choice((pyautogui.easeOutQuad, pyautogui.easeOutCirc))
                            else:
                                tween = random.choice((pyautogui.easeInOutQuad, pyautogui.easeInOutCirc))
                            pyautogui.moveTo(*point, duration=random.uniform(duration/5, duration*2), tween=tween)

                        pyHM.mouse.up()
                        export_button.click()


    if move_type == 2:  # PynPut
        for radius in [100, 200, 250, 300, 330]:
            for step in [15, 30, 50, 80, 100, 150, 250]:
                for duration in [.15, .1, .05, .01, .005, .001]:
                    for wait in [wait_linear, wait_quadratic, wait_gaussian]:
                        points = get_points_function(radius, step)
                        mouse = Controller()
                        mouse.position = points[0]
                        mouse.press(Button.left)

                        for i in range(1, len(points)):
                            mouse.position = points[i]
                            wait(duration, step, i)

                        mouse.release(Button.left)
                        export_button.click()

    elif move_type == 3:  # pyHM
        for radius in [100, 200, 250, 300, 330]:
            for step in [3, 7, 12, 15, 20, 25, 35, 45, 55, 65, 80]:
                points = get_points_function(radius, step)
                pyautogui.moveTo(*points[0])
                pyHM.mouse.down()

                for point in points[1:]:
                    try:
                        pyHM.mouse.move(*point, multiplier=1)
                    except ValueError:
                        # When two consecutive points are nearly the same...
                        pass

                pyHM.mouse.up()

                export_button.click()

    elif move_type == 4:  # PyClick (HumanClicker) based on Bezier curves
        for radius in [100, 200, 250, 300, 330]:
            for step in [15, 30, 50, 80, 100, 150, 250]:
                for duration in [.15, .1, .05, .01, .005, .001]:
                    points = get_points_function(radius, step)
                    hc = HumanClicker()
                    hc.move(points[0])
                    pyHM.mouse.down()

                    for i in range(1, len(points)):
                        hc.move(points[i], random.uniform(duration/5, duration*2))

                    pyHM.mouse.up()
                    export_button.click()

    elif move_type == 5:  # Java program with NaturalMouseMovement
        for radius in [280]:#[100, 200, 250]:
            for step in [3, 5, 8]:#[6, 13, 18, 30, 50, 80]:
                for motion in [0,1,2,3]:  # default, GrannyMotion, FastGamerMotion, AverageComputerUserMotion
                    points = get_points_function(radius, step)
                    pyautogui.moveTo(*points[0])
                    points.append(points[0])

                    pyHM.mouse.down()
                    # Need to put the screen scale to 100%, otherwise NaturalMouseMotion dosn't work
                    # The code of this jar is simply json parsing then factory.build(x, y).move();
                    subprocess.run(f"java -jar NaturalMouseMotionUsage-jar-with-dependencies.jar {motion} {json.dumps(points[1:])}")
                    pyHM.mouse.up()

                    export_button.click()


def main():
    driver = start_driver()

    time.sleep(.5)

    driver.switch_to.window(driver.window_handles[0])
    driver.close()
    driver.switch_to.window(driver.window_handles[4]) # before : 1


    time.sleep(.1)

    # With opera, sometimes a now window appears so we close it
    if driver.find_elements(By.ID, "cookie-consent"):
        driver.close()
        driver.switch_to.window(driver.window_handles[3])

    time.sleep(1)

    circle_automated(driver)

    time.sleep(3)

    driver.quit()

if __name__ == "__main__":
    main()
