import cv2
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--webcam', help="True/False", default=False)
parser.add_argument('--play_video', help="True/False", default=False)
parser.add_argument('--image', help="True/False", default=False)
parser.add_argument('--video_path', help="Path of video file", default="videos/car_on_road.mp4")
parser.add_argument('--image_path', help="Path of image to detect objects", default="Images/bicycle.jpg")
parser.add_argument('--verbose', help="To print statements", default=True)
args = parser.parse_args()


# Load yolo
def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3-608.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    print(colors)
    return net, classes, colors, output_layers


def load_image(img_path):
    # image loading
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    return img, height, width, channels


def start_webcam():
    cap = cv2.VideoCapture(0)

    return cap


def display_blob(blob):
    '''
        Three images each for RED, GREEN, BLUE channel
    '''
    for b in blob:
        for n, imgb in enumerate(b):
            cv2.imshow(str(n), imgb)


def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs


def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids


def draw_labels(boxes, confs, colors, class_ids, classes, img, M, pitch_pos):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    player_points = []

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            player_points.append(((x + w/2, y+h), color))
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)

    draw_pitch(img, pitch_pos)
    draw_players(player_points, img, M, pitch_pos)

    cv2.imshow("Image", img)


def draw_pitch(img, pitch_pos):
    pitch = cv2.imread('pitch.jpeg')
    pitch = cv2.cvtColor(pitch, cv2.COLOR_BGR2BGRA)

    img_h, img_w, img_c = img.shape
    overlay = np.zeros((img_h, img_w, 4), dtype='uint8')
    pitch_h, pitch_w, pitch_c = pitch.shape

    for i in range(0, pitch_h):
        for j in range(0, pitch_w):
            if pitch[i, j][3] != 0:
                overlay[i + pitch_pos[0], j + pitch_pos[1]] = pitch[i, j]
    cv2.addWeighted(overlay, 1, img, 1, 0, img)


def draw_players(player_points, img, M, pitch_pos):
    print(player_points)

    for point in player_points:
        pos, color = point
        pos = cv2.perspectiveTransform(np.float32([[pos]]), M)[0][0]
        print(pos)
        x, y = pos
        print(int(x))
        cv2.circle(img, (int(x) + pitch_pos[0], int(y) + pitch_pos[1]), radius=10, color=color, thickness=-1)


def image_detect(img_path):
    model, classes, colors, output_layers = load_yolo()
    image, height, width, channels = load_image(img_path)
    blob, outputs = detect_objects(image, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    draw_labels(boxes, confs, colors, class_ids, classes, image)
    while True:
        key = cv2.waitKey(1)
        if key == 27:
            break


def webcam_detect():
    model, classes, colors, output_layers = load_yolo()
    cap = start_webcam()
    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        draw_labels(boxes, confs, colors, class_ids, classes, frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()


def start_video(video_path):
    model, classes, colors, output_layers = load_yolo()
    cap = cv2.VideoCapture(video_path)
    print(cap)
    frame_count = 0
    frame_interval = 10

    input_pts = np.float32([[720, 450], [1186, 452], [160, 994], [1738, 1014]])
    output_pts = np.float32([[29, 289], [29, 101], [583, 268], [583, 101]])

    pitch_pos = (30, 30)

    # Compute the perspective transform M
    M = cv2.getPerspectiveTransform(input_pts, output_pts)

    while True:
        for i in range(9):
            temp = cap.grab()
        _, frame = cap.read()
        frame_count += frame_interval
        print(frame_count)

        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        print(outputs)

        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        draw_labels(boxes, confs, colors, class_ids, classes, frame, M, pitch_pos)

        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()


if __name__ == '__main__':
    webcam = args.webcam
    video_play = args.play_video
    image = args.image
    if webcam:
        if args.verbose:
            print('---- Starting Web Cam object detection ----')
        webcam_detect()
    if video_play:
        video_path = args.video_path
        if args.verbose:
            print('Opening ' + video_path + " .... ")
        start_video(video_path)
    if image:
        image_path = args.image_path
        if args.verbose:
            print("Opening " + image_path + " .... ")
        image_detect(image_path)

    cv2.destroyAllWindows()