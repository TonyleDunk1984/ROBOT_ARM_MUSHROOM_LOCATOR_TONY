import cv2
import numpy as np
import time
import csv
import socket
import os

from collections import deque
import math


# 1. The Setup (Global or Class variables)
# A deque is better than a list because it handles the 'pop' automatically
x_buffer = deque(maxlen=10)
y_buffer = deque(maxlen=10)

def update_position(raw_x, raw_y):
    # 2. Add new data
    x_buffer.append(raw_x)
    y_buffer.append(raw_y)
    
    # 3. Calculate Smooth Data
    world_x, world_y = pixel_to_world(u, v)
    print(f"Live world coordinates: x={world_x:.2f} cm, y={world_y:.2f} cm")
              # persist coordinates for robot consumption
    save_world_coords(world_x, world_y)
                # optionally send to robot controller (configure host/port)
                # send_coords_udp(world_x, world_y, host='192.168.1.2', port=5005)
    smooth_y = sum(y_buffer) / len(y_buffer)
    smooth_x = sum(x_buffer) / len(x_buffer)
    
    return smooth_x, smooth_y

center_x_pixel = 320  # example for 640x480 image
center_y_pixel = 240 # example for 640x480 image
pixel_to_world_scale_k = 0.005  # meters per pixel (tunable)
scale_factor_x = pixel_to_world_scale_k * 100  # cm per pixel
scale_factor_y = pixel_to_world_scale_k * 100  # cm per pixel
# The Math you are avoiding
def pixel_to_world(u, v):
    # u, v are the pixel coordinates of the mushroom center
    
    # OFFSET: subtract the center pixel (where the robot base is)
    # I think: center_x_pixel, center_y_pixel are the pixel coordinates of the robot base
    u_offset = u - center_x_pixel
    v_offset = v - center_y_pixel
    
    # SCALE: Convert to cm (flip signs if axes are opposed)
    # WARNING: Camera Y is usually "down" in pixels, World Y is "left/right".
    # Check your coordinate frames!
    world_x = u_offset * scale_factor_x 
    world_y = v_offset * scale_factor_y 
    
    return world_x, world_y


def save_world_coords(world_x, world_y, out_file='coords.csv', tag=None):
    """Append world coordinates to a CSV file with timestamp.

    Columns: timestamp, x_cm, y_cm, tag
    """
    header = ['timestamp', 'x_cm', 'y_cm', 'tag']
    write_header = not os.path.exists(out_file)
    try:
        with open(out_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow([time.time(), f"{world_x:.2f}", f"{world_y:.2f}", tag or ""])
        return out_file
    except Exception as e:
        print(f"Failed to save coords to {out_file}: {e}")
        return None


def send_coords_udp(world_x, world_y, host='192.168.1.2', port=5005, timeout=0.5):
    """Send coordinates as a UDP packet to robot controller. Returns True on success."""
    msg = f"{world_x:.2f},{world_y:.2f}".encode('utf-8')
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)
        sock.sendto(msg, (host, port))
        sock.close()
        return True
    except Exception as e:
        print(f"Failed to send UDP coords to {host}:{port}: {e}")
        return False

def detect_mushroom(image):
    """
    Detect a mushroom-like blob by color (brown or white).
    Returns (u, v, largest_contour) where (u,v) are image coordinates (ints).
    Returns None if no suitable contour found.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    

    # Brown-ish mask (tunable)
    brown_lower = np.array([5, 50, 20])
    brown_upper = np.array([30, 255, 200])
    mask_brown = cv2.inRange(hsv, brown_lower, brown_upper)

    # White mask
    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 40, 255])
    mask_white = cv2.inRange(hsv, white_lower, white_upper)

    mask = cv2.bitwise_or(mask_brown, mask_white)

    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours
    contours_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    if not contours:
        return None

    # Choose largest contour by area
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 100:  # reject too small blobs (tunable)
        return None

    M = cv2.moments(largest)
    if M.get("m00", 0) != 0:
        u = int(M["m10"] / M["m00"])
        v = int(M["m01"] / M["m00"])
    else:
        # fallback to contour center
        (u_f, v_f), _ = cv2.minEnclosingCircle(largest)
        u, v = int(u_f), int(v_f)

    return u, v, largest


def image_callback(camera_index=0):
    """
    Capture one frame from the laptop camera and return it (BGR).
    """
    cap = cv2.VideoCapture(0)  # or VideoCapture("C:/path/to/video.mp4")
    # optional warm-up: for _ in range(5): cap.read()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Blur the captured frame for noise reduction, then run detection on the blurred image
        blurred = cv2.GaussianBlur(frame, (11,11), 5,5)
        result = detect_mushroom(blurred) # pass blurred frame to the function
        if result is not None:
            u, v, cnt = result
            cv2.drawContours(frame, [cnt], -1, (255,0,0), 2, cv2.LINE_AA, 1, 0) # draw contour
            cv2.circle(frame, (u, v), 4, (0,0,255), -1) # draw center   
            # Print world coordinates to terminal for live feedback
            try:
                world_x, world_y = pixel_to_world(u, v)
                print(f"Live world coordinates: x={world_x:.2f} cm, y={world_y:.2f} cm")
                print(f"math.degrees(theta1) = {solve_ik(world_x, world_y)[0]:.2f} deg, math.degrees(theta2) = {solve_ik(world_x, world_y)[1]:.2f} deg")
            except Exception as e:
                print(f"Error converting to world coords: {e}")
        cv2.imshow("feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    # Return last frame captured (if any). If no frame was captured, raise.
    if 'frame' in locals() and frame is not None:
        return frame
    raise RuntimeError("No frame captured from camera")

def solve_ik(x, y, L1=20, L2=20): # Use your actual link lengths in cm
    # Calculate distance to target
    dist_sq = x**2 + y**2

    # Law of Cosines for theta2
    cos_t2 = (dist_sq - L1**2 - L2**2) / (2 * L1 * L2)
    # Boundary check to prevent crashes
    cos_t2 = max(-1, min(1, cos_t2)) 

    theta2 = math.acos(cos_t2) # Internal angle

    # Calculate theta1
    alpha = math.atan2(y, x)
    beta = math.acos((dist_sq + L1**2 - L2**2) / (2 * L1 * math.sqrt(dist_sq)))

    theta1 = alpha - beta # Elbow-up solution

    return math.degrees(theta1), math.degrees(theta2)

if __name__ == "__main__":
    img = image_callback(0)
    result = detect_mushroom(img)
    if result is None:
        print("No mushroom-like object detected.")
    else:
        u, v, cnt = result
        print(f"Detected center: (u={u}, v={v})")
        # Convert pixel coordinates to world coordinates and print them
        world_x, world_y = pixel_to_world(u, v)
        print(f"World coordinates: x={world_x:.2f} cm, y={world_y:.2f} cm")
        # visualize
        out = img.copy()
        cv2.drawContours(out, [cnt], -1, (0,255,0), 2)
        cv2.circle(out, (u, v), 4, (0,0,255), -1)
        cv2.imshow("Detection", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
