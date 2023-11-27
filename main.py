import cv2
import numpy as np
import imutils
import screeninfo
import pyautogui


def get_center_of_hand(image):
    # convert BGR to HSL(hue, saturation, lightness)
    image_hsl = cv2.cvtColor(image, cv2.COLOR_BGR2HLS_FULL)
    cv2.imshow("HSL", image_hsl)

    # lower and upper HSL bounds to be used on image_hsl
    lower_bound = np.array([0, 38, 38])
    upper_bound = np.array([18, 150, 250])

    # Checks if each image_hsl pixel is within the lower and upper hsl bounds
    skin_tone_binary = cv2.inRange(image_hsl, lower_bound, upper_bound)
    cv2.imshow("binary", skin_tone_binary)

    # find the contours using the extracted hand image
    contours = cv2.findContours(skin_tone_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # simplifies handling of contours
    contours = imutils.grab_contours(contours)

    # Sort contours so the largest contour is first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Get the largest contour (hopefully the hand)
    largest_contour = contours[0]

    # Get the convex hull for the largest contour
    convex_hull = [cv2.convexHull(largest_contour)]

    # create an empty black image (displaying image only)
    # Draw contours on frame image
    cv2.drawContours(image, contours[0], -1, (169, 107, 255), 5)

    # get image moments from convex hull
    moments = cv2.moments(convex_hull[0])

    # calculate the center of the convex hull/hopefully the hand
    convex_center_x = int(moments["m10"] / moments["m00"])
    convex_center_y = int(moments["m01"] / moments["m00"])

    # draw the largest contour/convex hull and a circle at the center of the convex hull
    image_convexHull = np.zeros((skin_tone_binary.shape[0], skin_tone_binary.shape[1], 3))
    cv2.drawContours(image_convexHull, convex_hull, 0, (255, 0, 0), 2, 8)
    cv2.drawContours(image_convexHull, contours, 0, (0, 255, 0), 2, 8)
    cv2.circle(image_convexHull, (convex_center_x, convex_center_y), 5, (0, 0, 255), 2, 7)

    cv2.imshow("contour,convex,center", image_convexHull)
    return convex_center_x, convex_center_y


def convert_center_pos_to_screen_pos(c_x, c_y, frame_size):
    # Get the width and height of the primary monitor
    primary_monitor = screeninfo.get_monitors()[0]
    screen_width = primary_monitor.width
    screen_height = primary_monitor.height

    # calculate ratio between frame and window screen
    x_ratio = screen_width / frame_size[1]  # frame's width
    y_ratio = screen_height / frame_size[0]  # frame's height

    # return the proper mouse coordinates
    return c_x * x_ratio, c_y * y_ratio


if __name__ == "__main__":
    camera = cv2.VideoCapture(0)  # Captures video from camera 0

    # Continuously read and process the next frame
    while True:
        return_val, frame = camera.read()
        if return_val:
            frame = cv2.flip(frame, 1)
            center_x, center_y = get_center_of_hand(frame)
            screen_x, screen_y = convert_center_pos_to_screen_pos(center_x, center_y, frame.shape)
            pyautogui.moveTo(screen_x, screen_y, .01)

        # End program if space key is pressed
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    camera.release()
    cv2.destroyAllWindows()
