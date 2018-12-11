import cv2 as cv
import argparse
import numpy as np
import os.path
import sys
import random

# Initialize the parameters
confThreshold = 0.6  # Confidence threshold
maskThreshold = 0.3  # Mask threshold
# confThreshold = 0.5  # Confidence threshold
# maskThreshold = 0.3  # Mask threshold

parser = argparse.ArgumentParser(description='Use this script to run Mask-RCNN object detection and segmentation')
parser.add_argument('--image', help='Path to image file')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

# speed-up using multithreads
cv.setUseOptimized(True)
cv.setNumThreads(4)



# Draw the predicted bounding box, colorize and show the mask on the image
def drawBox(frame, classId, conf, left, top, right, bottom, classMask):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    # Print a label of class.
    label = '%.2f' % conf
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                 (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

    # Resize the mask, threshold, color and apply it on the image
    classMask = cv.resize(classMask, (right - left + 1, bottom - top + 1))
    # print("classMask :", classMask)
    mask = (classMask > maskThreshold)

    roi = frame[top:bottom + 1, left:right + 1][mask]

    # color = colors[classId%len(colors)]
    # Comment the above line and uncomment the two lines below to generate different instance colors
    colorIndex = random.randint(0, len(colors) - 1)
    color = colors[colorIndex]

    frame[top:bottom + 1, left:right + 1][mask] = ([0.3 * color[0], 0.3 * color[1], 0.3 * color[2]] + 0.7 * roi).astype(np.uint8)

    # Draw the contours on the image
    mask = mask.astype(np.uint8)
    im2, contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cv.drawContours(frame[top:bottom + 1, left:right + 1], contours, -1, color, 3, cv.LINE_8, hierarchy, 100)



def drawBox_each(frame, classId, conf, left, top, right, bottom, classMask, numDetections):
    """
    :param frame:
    :param classId:
    :param conf:
    :param left:
    :param top:
    :param right:
    :param bottom:
    :param classMask:
    :param numDetections:
    :return: detected(cropped) each image
    """
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    # Print a label of class.
    label = '%.2f' % conf
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                 (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

    # Resize the mask, threshold, color and apply it on the image
    classMask = cv.resize(classMask, (right - left + 1, bottom - top + 1))
    # print("classMask :", classMask)
    mask = (classMask > maskThreshold)

    roi = frame[top:bottom + 1, left:right + 1][mask]

    # color = colors[classId%len(colors)]
    # Comment the above line and uncomment the two lines below to generate different instance colors
    colorIndex = random.randint(0, len(colors) - 1)
    color = colors[colorIndex]

    frame[top:bottom + 1, left:right + 1][mask] = ([0.3 * color[0], 0.3 * color[1], 0.3 * color[2]] + 0.7 * roi).astype(np.uint8)

    # Draw the contours on the image
    mask = mask.astype(np.uint8)
    im2, contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # remove the contours from the image and show the resulting image
    masked_img = cv.bitwise_and(frame[top:bottom + 1, left:right + 1], frame[top:bottom + 1, left:right + 1], mask=mask)
    cv.imwrite("./detected_img/" + str(numDetections) + "_masked.jpg", masked_img)

    cv.drawContours(frame[top:bottom + 1, left:right + 1], contours, -1, color, 3, cv.LINE_8, hierarchy, 100)
    cv.imwrite("./detected_img/"+str(numDetections)+"_roi.jpg", frame[top:bottom + 1, left:right + 1])


def drawBox_each_file_name(frame, classId, conf, left, top, right, bottom, classMask, numDetections, file_name):
    """
    :param frame:
    :param classId:
    :param conf:
    :param left:
    :param top:
    :param right:
    :param bottom:
    :param classMask:
    :param numDetections:
    :return: detected(cropped) each image
    """
    # Draw a bounding box.
    # cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    # Print a label of class.
    label = '%.2f' % conf
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
    #              (255, 255, 255), cv.FILLED)
    # cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

    # Resize the mask, threshold, color and apply it on the image
    classMask = cv.resize(classMask, (right - left + 1, bottom - top + 1))
    # print("classMask :", classMask)
    mask = (classMask > maskThreshold)

    roi = frame[top:bottom + 1, left:right + 1][mask]

    # color = colors[classId%len(colors)]
    # Comment the above line and uncomment the two lines below to generate different instance colors
    # colorIndex = random.randint(0, len(colors) - 1)
    # color = colors[colorIndex]
    #
    # frame[top:bottom + 1, left:right + 1][mask] = ([0.3 * color[0], 0.3 * color[1], 0.3 * color[2]] + 0.7 * roi).astype(np.uint8)

    # Draw the contours on the image
    mask = mask.astype(np.uint8)
    im2, contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # remove the contours from the image and show the resulting image
    masked_img = cv.bitwise_and(frame[top:bottom + 1, left:right + 1], frame[top:bottom + 1, left:right + 1], mask=mask)
    cv.imwrite("./detected_img/" + file_name.split("/")[2]+"_"+str(numDetections) + "_masked.jpg", masked_img)
    print("file_name :", file_name.split("/")[-1])


    # cv.drawContours(frame[top:bottom + 1, left:right + 1], contours, -1, color, 3, cv.LINE_8, hierarchy, 100)
    # cv.imwrite(file_name+"_"+str(numDetections)+"_roi.jpg", frame[top:bottom + 1, left:right + 1])



# For each frame, extract the bounding box and mask for each detected object
def postprocess(boxes, masks):
    # Output size of masks is NxCxHxW where
    # N - number of detected boxes
    # C - number of classes (excluding background)
    # HxW - segmentation shape

    numClasses = masks.shape[1] # 90
    numDetections = boxes.shape[2] # 8


    frameH = frame.shape[0]
    frameW = frame.shape[1]

    for i in range(numDetections):
        box = boxes[0, 0, i] # boxes :  (1, 1, 8, 7) , box : (7, )
        # print("box.shape : ", box.shape)
        # print("box :, ", box)
        mask = masks[i]
        score = box[2]

        # print("box[2].shape : ", box[2].shape)
        if score > confThreshold:
            classId = int(box[1])
            print("classId :", classId)

            # Extract the bounding box
            left = int(frameW * box[3])
            top = int(frameH * box[4])
            right = int(frameW * box[5])
            bottom = int(frameH * box[6])

            left = max(0, min(left, frameW - 1))
            top = max(0, min(top, frameH - 1))
            right = max(0, min(right, frameW - 1))
            bottom = max(0, min(bottom, frameH - 1))

            # Extract the mask for the object
            classMask = mask[classId]

            # Draw bounding box, colorize and show the mask on the image
            # drawBox(frame, classId, score, left, top, right, bottom, classMask)
            drawBox_each(frame, classId, score, left, top, right, bottom, classMask, i)

# For each frame, extract the bounding box and mask for each detected object
def postprocess_each(boxes, masks, file_name):
    # Output size of masks is NxCxHxW where
    # N - number of detected boxes
    # C - number of classes (excluding background)
    # HxW - segmentation shape

    numClasses = masks.shape[1] # 90
    numDetections = boxes.shape[2] # 8


    frameH = frame.shape[0]
    frameW = frame.shape[1]

    for i in range(numDetections):
        box = boxes[0, 0, i] # boxes :  (1, 1, 8, 7) , box : (7, )
        # print("box.shape : ", box.shape)
        # print("box :, ", box)
        mask = masks[i]
        score = box[2]

        # print("box[2].shape : ", box[2].shape)
        if score > confThreshold:
            classId = int(box[1])
            print("classId :", classId)

            # Extract the bounding box
            left = int(frameW * box[3])
            top = int(frameH * box[4])
            right = int(frameW * box[5])
            bottom = int(frameH * box[6])

            left = max(0, min(left, frameW - 1))
            top = max(0, min(top, frameH - 1))
            right = max(0, min(right, frameW - 1))
            bottom = max(0, min(bottom, frameH - 1))

            # Extract the mask for the object
            classMask = mask[classId]

            # Draw bounding box, colorize and show the mask on the image
            # drawBox(frame, classId, score, left, top, right, bottom, classMask)
            drawBox_each_file_name(frame, classId, score, left, top, right, bottom, classMask, i, file_name)





# Load names of classes
classesFile = "mscoco_labels_for_cloth.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the textGraph and weight files for the model
textGraph = "./mask_rcnn_inception_v2_coco_2018_01_28/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
modelWeights = "./mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"

# Load the network
net = cv.dnn.readNetFromTensorflow(modelWeights, textGraph)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Load the classes
colorsFile = "colors.txt"
with open(colorsFile, 'rt') as f:
    colorsStr = f.read().rstrip('\n').split('\n')
colors = []  # [0,0,0]
for i in range(len(colorsStr)):
    rgb = colorsStr[i].split(' ')
    color = np.array([float(rgb[0]), float(rgb[1]), float(rgb[2])])
    colors.append(color)

# winName = 'Mask-RCNN Object detection and Segmentation in OpenCV'
winName = ''
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "mask_rcnn_out_py.avi"



if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4] + '_mask_rcnn_out_py.jpg'

elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4] + '_mask_rcnn_out_py.avi'
else:
    # Webcam input
    cap = cv.VideoCapture(0)

# Get the video writer initialized to save the output video
if (not args.image):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 28,
                                (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

while cv.waitKey(1) < 0:

    # Get frame from the video
    hasFrame, frame = cap.read()

    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        break

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, swapRB=True, crop=False) # blob (1, 3, 720, 1280) (NxCxHxW)

    # Set the input to the network
    net.setInput(blob)

    # Run the forward pass to get output from the output layers
    boxes, masks = net.forward(['detection_out_final', 'detection_masks'])
    # boxes.shape : (1, 1, 8, 7)
    # masks.shape : (100, 90, 15, 15)

    # Extract the bounding box and mask for each of the detected objects
    # postprocess(boxes, masks, args.image[:-4])
    postprocess_each(boxes, masks, args.image[:-4])
    # Put efficiency information.
    t, _ = net.getPerfProfile()
    label = 'Mask-RCNN on 2.8 GHz Intel Core i7 CPU, Inference time for a frame : %0.0f ms' % abs(
        t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # Write the frame with the detection boxes
    if (args.image):
        cv.imwrite(outputFile, frame.astype(np.uint8))
        # cv.imwrite(detectedFile, boxes.astype(np.uint8))
    else:
        vid_writer.write(frame.astype(np.uint8))


    # to show cam image
    # cv.imshow(winName, frame)