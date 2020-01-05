import os
import argparse
import cv2
import freenect
import numpy

# Command line arguments
parser = argparse.ArgumentParser(description='Kinect video recorder')
parser.add_argument("-colour-path", default="/media/dmytro/Storage/Ubuntu/Documents/Projects/Kinect/colour", help="location of colour frames")
parser.add_argument("-depth-path", default="/media/dmytro/Storage/Ubuntu/Documents/Projects/Kinect/depth", help="location of depth frames")

# Windows for display
cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)

def record():
    # A list of captured rgbd frames
    colour = list()
    depth = list()

    # Capture loop
    capture = False
    running = True
    frameCount = 1

    while running:
        rgb_image = freenect.sync_get_video(index=0)[0]
        depth_image = freenect.sync_get_depth(index=0, format=freenect.DEPTH_REGISTERED)[0]

        cv2.imshow("RGB", cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        cv2.imshow("Depth", depth_image * 20)

        if capture:
            print("Frame %08d" % frameCount)
            colour.append(numpy.copy(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)))
            depth.append(numpy.copy(depth_image))
            frameCount = frameCount + 1

        key = cv2.waitKey(1)

        if key == ord(" "):
            capture = not capture
        elif key == ord("x"):
            running = False

    return colour, depth

if __name__ == "__main__":
    args = parser.parse_args()
    colour, depth = record()

    if not os.path.exists(args.colour_path):
        os.mkdir(args.colour_path)
    if not os.path.exists(args.depth_path):
        os.mkdir(args.depth_path)

    for i in range(len(colour)):
        filename = "/%08d" % i
        print(filename)
        cv2.imwrite("".join([args.colour_path, filename, ".png"]), colour[i])
        numpy.save("".join([args.depth_path, filename, ".npy"]), depth[i])
