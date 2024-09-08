import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import threading

def run_testing():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    classifier = Classifier("model1/keras_model.h5", "model1/labels.txt")
    offset = 20
    imgSize = 300
    labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
    # labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

    def process_frame():
        while True:
            success, img = cap.read()
            if not success:
                break
            imgOutput = img.copy()
            hands, img = detector.findHands(img)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

                # Get text size for dynamic rectangle
                text = labels[index]
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1.7, 2)

                # Add padding around the text
                padding = 10
                rectangle_width = text_width + 2 * padding
                rectangle_height = text_height + 2 * padding

                # Draw a dynamic rectangle that fits the text
                cv2.rectangle(imgOutput, (x - offset, y - offset - rectangle_height - 10),  # Top-left corner
                              (x - offset + rectangle_width, y - offset - 10),  # Bottom-right corner
                              (255, 0, 255), cv2.FILLED)  # Color (pink) and fill the rectangle

                # Display the label text inside the rectangle
                cv2.putText(imgOutput, text, (x - offset + padding, y - offset - padding - baseline),  # Position for the text label
                            cv2.FONT_HERSHEY_COMPLEX, 1.7,  # Font and size
                            (255, 255, 255), 2)  # Color (white) and thickness of the text

                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (255, 0, 255), 4)

                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageWhite", imgWhite)

            cv2.imshow("Image", imgOutput)
            key = cv2.waitKey(1)
            if key == ord('q'):  # Press 'q' to exit the loop
                break

    try:
        # Run the frame processing in a separate thread
        thread = threading.Thread(target=process_frame)
        thread.start()
        thread.join()  # Ensure the thread completes
    finally:
        # cap.release()
        cv2.destroyAllWindows()

    return "Testing completed successfully."

def run_testing_words():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    classifier = Classifier("Model2/keras_model.h5", "Model2/labels.txt")
    offset = 20
    imgSize = 300
    labels = ["Bye", "Hello", "No", "Perfect", "Thank You", "Yes"]

    def process_frame():
        while True:
            success, img = cap.read()
            if not success:
                break
            imgOutput = img.copy()
            hands, img = detector.findHands(img)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

                # Get text size for dynamic rectangle
                text = labels[index]
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1.7, 2)

                # Add padding around the text
                padding = 10
                rectangle_width = text_width + 2 * padding
                rectangle_height = text_height + 2 * padding

                # Draw a dynamic rectangle that fits the text
                cv2.rectangle(imgOutput, (x - offset, y - offset - rectangle_height - 10),  # Top-left corner
                              (x - offset + rectangle_width, y - offset - 10),  # Bottom-right corner
                              (255, 0, 255), cv2.FILLED)  # Color (pink) and fill the rectangle

                # Display the label text inside the rectangle
                cv2.putText(imgOutput, text, (x - offset + padding, y - offset - padding - baseline),  # Position for the text label
                            cv2.FONT_HERSHEY_COMPLEX, 1.7,  # Font and size
                            (255, 255, 255), 2)  # Color (white) and thickness of the text

                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (255, 0, 255), 4)

                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageWhite", imgWhite)

            cv2.imshow("Image", imgOutput)
            key = cv2.waitKey(1)      
            if key == ord('q'):  # Press 'q' to exit the loop
                break

    try:
        # Run the frame processing in a separate thread
        thread = threading.Thread(target=process_frame)
        thread.start()
        thread.join()  # Ensure the thread completes
    finally:
        # cap.release()
        cv2.destroyAllWindows()

    return "Testing completed successfully."

import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
import threading

# Function to draw Gujarati text
def draw_gujarati_text(img, text, position, font_path='NotoSansGujarati-Black.ttf', font_size=32, color=(255, 255, 255)):
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(font_path, font_size)
        draw.text(position, text, font=font, fill=color)
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error rendering Gujarati text: {e}")
    return img

# Process the video stream for Gujarati sign language letters
def gujrati_text():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    classifier = Classifier("Model5/keras_model.h5", "Model5/labels.txt")
    offset = 20
    imgSize = 300
    labels = ["૧", "૨", "૩", "૪","ઇ"]  # Gujarati labels

    def process_frame():
        while True:
            success, img = cap.read()
            if not success:
                break
            imgOutput = img.copy()
            hands, img = detector.findHands(img)

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                # Predict the gesture
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

                # Get the label text in Gujarati
                text = labels[index]

                # Calculate text size and position dynamically
                padding = 10
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1.7, 2)
                rectangle_width = text_width + 2 * padding
                rectangle_height = text_height + 2 * padding

                # Draw the dynamic rectangle that fits the Gujarati text
                cv2.rectangle(imgOutput, (x - offset, y - offset - rectangle_height - 10),  # Top-left corner
                              (x - offset + rectangle_width, y - offset - 10),  # Bottom-right corner
                              (255, 0, 255), cv2.FILLED)  # Color (pink) and fill the rectangle

                # Render the Gujarati text inside the rectangle using PIL
                imgOutput = draw_gujarati_text(imgOutput, text, (x - offset + padding, y - offset - rectangle_height - padding))

                # Draw a rectangle around the detected hand
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (255, 0, 255), 4)

                # Show the processed images
                cv2.imshow("ImageWhite", imgWhite)

            # Display the output with the Gujarati text
            cv2.imshow("Image", imgOutput)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    try:
        # Run the frame processing in a separate thread
        thread = threading.Thread(target=process_frame)
        thread.start()
        thread.join()  # Ensure the thread completes
    finally:
        # cap.release()
        cv2.destroyAllWindows()

    return "Gujarati Testing completed successfully."

# Now call the gujrati_text function
# gujrati_text()
