import logging
import airsim
import time
import math
import os
import csv
import random
import keyboard
import threading
import pygame
import numpy as np
from datetime import datetime
from pythonosc import dispatcher, osc_server
import subprocess
import signal
from plyer import notification

client_lock = threading.Lock()

class ReusableOSCUDPServer(osc_server.ThreadingOSCUDPServer):
    allow_reuse_address = True


#Pop Up for Adaptive Automation
def show_message(title, message, duration=3):
    notification.notify(
        title=title,
        message=message,
        timeout=duration
        )

webcam_name = "Logi Webcam C920e"
ffmpeg_process = None

def start_ffmpeg_recording(output_path, camera_name = "Logi Webcam C920e"):
    global ffmpeg_process

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",                        
        "-f", "dshow",
        "-framerate", "30",
        "-video_size", "1920x1080",
        "-i", f"video={camera_name}",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        output_path
    ]

    print("Starting FFmpeg recording...")
    ffmpeg_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.PIPE)

def stop_ffmpeg_recording():
    global ffmpeg_process
    if ffmpeg_process:
        print("Stopping FFmpeg recording...")
        try:
            ffmpeg_process.stdin.write(b"q")    #Attempt graceful close
            ffmpeg_process.stdin.flush()
        except Exception:
            ffmpeg_process.terminate()
        
        ffmpeg_process.wait()
        ffmpeg_process = None

a = 30

BASE_DATA_DIR = "trial_data"
os.makedirs(BASE_DATA_DIR, exist_ok=True)

trial_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
trial_folder = os.path.join(BASE_DATA_DIR, f"trial_{trial_timestamp}")
os.makedirs(trial_folder, exist_ok=True)

csv_file = os.path.join(trial_folder, f"biosignals_{trial_timestamp}.csv")
video_file = os.path.join(trial_folder, f"webcam_{trial_timestamp}.mp4")
log_filename = os.path.join(trial_folder, f"collision_log_{trial_timestamp}.log")

with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "variable", "value"])

def log_measurement(variable, value):
    timestamp = datetime.now().isoformat()
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, variable, value])

user_input = input("Please enter a number between 1 and 5: ")

user_number = int(user_input)

latest = {"eda": None, "hr": None}
baseline_data = {"eda": [], "hr": []}
baseline_collected = False
baseline_duration = 15

start_time = None

def get_first_value(eda):
    if isinstance(eda, (list, tuple)):
        return eda[0] if eda else None
    return eda

def eda(addr, *val):
    global start_time
    first_value = get_first_value(val)
    latest["eda"] = float(first_value)
    log_measurement("EDA", val)
    if not baseline_collected:
        baseline_data["eda"].append(float(first_value))
        if start_time is None: start_time = time.time()

def hr(addr, val):
    latest["hr"] = float(val)
    log_measurement("HR", val)
    if not baseline_collected:
        baseline_data["hr"].append(float(val))

disp = dispatcher.Dispatcher()
disp.map("/EmotiBit/0/EDA", eda)
disp.map("/EmotiBit/0/HR", hr)


def start_server():
    server = ReusableOSCUDPServer(("0.0.0.0", 12345), disp)
    print("OSC server running...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down server...")
    finally:
        server.shutdown()
        server.server_close()
        
threading.Thread(target=start_server, daemon=True).start()

baseline = {}

def compute_stress():
    hr_marker = False
    eda_marker = False
    global baseline, baseline_collected

    if not baseline_collected:
        if start_time and ((time.time() - start_time) >= baseline_duration):
            baseline = {
                "eda_mean":  np.mean(baseline_data["eda"]),
                "hr_mean":   np.mean(baseline_data["hr"]),
            }
            baseline_collected = True
            print("Baseline Collected")
        return hr_marker, eda_marker

    if None in latest.values():
        return hr_marker, eda_marker # not enough data yet

    z_eda  = abs(latest["eda"] - baseline["eda_mean"])
    z_hr   = (latest["hr"]   - baseline["hr_mean"])

    #print(z_eda)
    #print(z_hr)

    if z_hr >= 9:
        hr_marker = True
    else:
        hr_marker = False

    if z_eda >= 0.007:
        eda_marker = True
    else:
        eda_marker = False

    return hr_marker, eda_marker

def Stress():
    global a
    gap = 5
    while True:
        hrMarker, edaMarker = compute_stress()

        if user_number == 1:
            if hrMarker == True:
                a = 90
                time.sleep(gap)
            else:
                a = 30
                time.sleep(gap)

        if user_number == 2:
            if edaMarker == True:
                a = 90
                time.sleep(gap)
            else:
                a = 30
                time.sleep(gap)

        if user_number == 3:
            if hrMarker == True and edaMarker == True:
                a = 90
                time.sleep(gap)
            else:
                a = 30
                time.sleep(gap)

        if user_number == 4:
            if hrMarker == True or edaMarker == True:
                a = 90
                time.sleep(gap)
            else:
                a = 30
                time.sleep(gap)

        if user_number == 5:
            time.sleep(gap)
            continue
       



stress_thread = threading.Thread(target=Stress)
stress_thread.daemon = True
stress_thread.start()

def toggle_a():
    global a
    with client_lock:
        a = 30 if a == 90 else 90
    print(f"Toggled 'a' to: {a}")

keyboard.add_hotkey('space', toggle_a)

'''
#Put Logs in folder "logs"
log_dir = "logs"

date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = os.path.join(log_dir, f"collision_log_{date}.log")
'''

#Log config
logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        filemode='w'
    )

#Connect to airsim
client = airsim.MultirotorClient()
client.confirmConnection()

print("Connected to AirSim.")

#Pygame Mixer
pygame.mixer.init()

sound = pygame.mixer.Sound("censor-beep-102309.mp3")
sound.set_volume(1.0)


#Clamp Function
def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

#Changes wind every update_interval by a random value between -1 and 1
def wind(update_interval, max_wind):
    global a
    AA = False
    thread_client = airsim.MultirotorClient()
    thread_client.confirmConnection()

    wind = [0.0, 0.0, 0.0]

    for i in range(2):
                delta = random.uniform((0.35*max_wind), (0.65*max_wind))
                sign = random.choice([-1, 1])

                delta = sign * delta
                wind[i] = delta

    while True:
        with client_lock:
            a1 = a
        if a1 == 30:
            for i in range(2):
                delta = random.uniform(-1.0, 1.0)
                wind[i] += delta
                wind[i] = clamp(wind[i], -max_wind, max_wind)

            wind_vector = airsim.Vector3r(wind[0], wind[1], wind[2])
            with client_lock:
                thread_client.simSetWind(wind_vector)
            time.sleep(update_interval)
        elif a1 == 90:
            wind_vector = airsim.Vector3r(0, 0, 0)
            with client_lock:
                thread_client.simSetWind(wind_vector)
            time.sleep(update_interval)
       
def getClosestTarget(xVal, yVal, zVal, visited):

    closestDistance = float('inf')
    closestPose = (0, 0, 0)
    closestTarget = None

    for i in range(28):
        if i in visited:
            continue

        AirsimPose = client.simGetObjectPose("Target_Point_" + str(i))

        unreal_position = (
            (AirsimPose.position.x_val * 100) + 3000,
            (AirsimPose.position.y_val * 100) - 4000,
            -(AirsimPose.position.z_val * 100) + 200
        )

        distance = math.sqrt(
            (unreal_position[0] - xVal) ** 2 +
            (unreal_position[1] - yVal) ** 2 +
            (unreal_position[2] - zVal) ** 2
        )

        if distance < closestDistance:
            closestPose = unreal_position
            closestTarget = i
            closestDistance = distance
            TargetPose = AirsimPose
   
    return closestPose, closestTarget, TargetPose

def calculate_yaw(current_pos, target_pos):
    dx = target_pos.position.x_val - current_pos.position.x_val
    dy = target_pos.position.y_val - current_pos.position.y_val
    yaw = math.degrees(math.atan2(dy, dx))
    return yaw

#Collisions and timing
def main():

    api_control_enabled = False
    airsim_position = None

    visited_targets = set()
   
    start_time = -1
    end_time = -2
    elapsed = -3
   
    Point = 50

    interval = 0.25

    start = False

    out_of_bounds = 0
    in_bounds = 0

    Collision_Number = -1
    Auto_Collision_Number = -1

    rotated_for_target = False

    wind_thread = threading.Thread(target=wind, kwargs={'update_interval': 1, 'max_wind': 10})
    wind_thread.daemon = True
    wind_thread.start()

    AA = False

    try:
        while True:
            with client_lock:
               
                state = client.getMultirotorState()

                collision_info = client.simGetCollisionInfo()

                pose = client.simGetVehiclePose()

                unreal_position = (
                    (pose.position.x_val * 100) + 3000,
                    (pose.position.y_val * 100) - 4000,
                    -(pose.position.z_val * 100) + 200
                )  

                if (a > 60 and start):

                    if AA == False:
                        show_message("Adaptive Automation", "Automation On", duration=2)
                        AA = True

                    if not api_control_enabled:
                        client.enableApiControl(True)
                        api_control_enabled = True
                        client.takeoffAsync().join()

                    if state.landed_state == airsim.LandedState.Landed:
                        client.takeoffAsync().join()
                        time.sleep(2)

                    else:
                        closestPose, ClosestTarget, AirsimPose = getClosestTarget(
                            xVal=unreal_position[0],
                            yVal=unreal_position[1],
                            zVal=unreal_position[2],
                            visited=visited_targets
                        )

                        if not rotated_for_target:
                            yaw = calculate_yaw(current_pos=pose, target_pos=AirsimPose)
                            client.rotateToYawAsync(yaw).join()
                            rotated_for_target = True


                        airsim_position = (
                            (closestPose[0] - 3000) / 100,
                            (closestPose[1] + 4000) / 100,
                            -(closestPose[2] - 200) / 100
                        )

                        target_position = airsim.Vector3r(*airsim_position)
                        Point = ClosestTarget
                        client.moveToPositionAsync(
                            target_position.x_val,
                            target_position.y_val,
                            target_position.z_val,
                            velocity=2,
                            lookahead=-1,
                            adaptive_lookahead=0,
                            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom
                        )
                        print("Moving to "+ str(Point))
                        print(f"Moving to target position: {target_position.x_val}, {target_position.y_val}, {target_position.z_val}")


                        # Check if drone is near target
                        current_pos = state.kinematics_estimated.position
                        dist = math.sqrt(
                            (current_pos.x_val - airsim_position[0]) ** 2 +
                            (current_pos.y_val - airsim_position[1]) ** 2    
                        )
                        print(dist)

                        if dist < 1:
                            print("Reached target.")
                            rotated_for_target = False
                            client.hoverAsync()
                            visited_targets.update(range(Point + 1))
                            print("added " + str(Point) + " to list")

                else:

                    if AA == True:
                        show_message("Adaptive Automation", "Automation Off", duration=2)
                        AA = False

                    if api_control_enabled:
                        client.enableApiControl(False)
                        api_control_enabled = False



                if (((-3400 > unreal_position[1] > -4500) and (7280 > unreal_position[0] > 3300) and unreal_position[2] < 1300)):
                    in_bounds += interval
                    sound.stop()

                elif((-690 < unreal_position[1] < 40) and unreal_position[2] < 405):
                    in_bounds += interval
                    sound.stop()

                elif ((7280 > unreal_position[0] > 6650) and (-4700 < unreal_position[1] < 150) and 150 < unreal_position[2] < 1300):
                    in_bounds += interval
                    sound.stop()

                else:
                    out_of_bounds += interval
                    if start:
                        sound.play()
                    #print("Out of Bounds")


                if collision_info.has_collided:
                    #Check for start
                    if (collision_info.object_name == "Finish1_Blueprint" and start_time == -1):
                        start_time = time.time()
                        print("Starting Webcam")
                        start_ffmpeg_recording(video_file, webcam_name)
                        out_of_bounds = 0
                        in_bounds = 0
                        print("Stopwatch Started")
                        Collision_Number = 0
                        Auto_Collision_Number = 0
                        start = True
                        logging.info("Start\n")

                    #Check for stop
                    elif (collision_info.object_name == "Finish1_Blueprint2" and start_time != -1 and end_time == -2):
                        stop_ffmpeg_recording()
                        end_time = time.time()
                        elapsed = end_time - start_time
                        print("Stopwatch Stopped")

                        logging.info("End")
                        logging.info("Wind: Yes")
                        logging.info("Number of Collisions: " + str(Collision_Number))
                        logging.info("Number of Auto Collisions: " + str(Auto_Collision_Number))
                        logging.info("Time: " + str(elapsed))
                        logging.info("Out of Bounds Time: " + str(out_of_bounds))
                        sound.stop()
                        break

                    #Print Collisions
                    else:
                        if a <= 60:
                            msg = (f"Collision \nObject: {collision_info.object_name}, "
                                f"Impact Point: {collision_info.impact_point}"
                                f"Normal: {collision_info.normal}"
                                f"Penetration: {collision_info.penetration_depth}"
                                "\n")
                            #print(msg)
                            logging.info(msg)
                            Collision_Number += 1
                        else:
                            msg = (f"Auto-flight collision with {collision_info.object_name} ignored.")
                            #print(msg)
                            logging.info(msg)
                            Auto_Collision_Number += 1


                time.sleep(interval)

    except KeyboardInterrupt:
        logging.info("Wind: Yes")
        logging.info("Number of Collisions: " + str(Collision_Number))
        logging.info("Number of Auto Collisions: " + str(Auto_Collision_Number))
        logging.info("Time: " + str(elapsed))
        logging.info("Out of Bounds Time: " + str(out_of_bounds))
        sound.stop()
        print("\nStopped checking for collisions.")



if __name__ == "__main__":
    main()

