import cv2
import numpy as np

def image_callback(camera_index=0):
    """
    Capture one frame from the laptop camera and return it (BGR).
    """
    cap = cv2.imread("C:/readyourimagepath")
    if not cap is None:
        return cap
    raise RuntimeError("Failed to read image from file")
    return frame

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

if __name__ == "__main__":
    img = image_callback(0)
    result = detect_mushroom(img)
    if result is None:
        print("No mushroom-like object detected.")
    else:
        u, v, cnt = result
        print(f"Detected center: (u={u}, v={v})")
        # visualize
        out = img.copy()
        cv2.drawContours(out, [cnt], -1, (0,255,0), 2)
        cv2.circle(out, (u, v), 4, (0,0,255), -1)
        cv2.imshow("Detection", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()