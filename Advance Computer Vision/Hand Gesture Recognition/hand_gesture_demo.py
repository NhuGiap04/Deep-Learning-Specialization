import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

class_name = 'None'


# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global class_name
    print('gesture recognition result: {}'.format(result))
    if len(result.gestures) > 0:
        class_name = result.gestures[0][0].category_name
    else:
        class_name = 'None'


options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

recognizer = GestureRecognizer.create_from_options(options)

# For webcam input:
with GestureRecognizer.create_from_options(options) as recognizer:
    vid = cv2.VideoCapture(0)
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while True:
            # Use OpenCV’s VideoCapture to start capturing from the webcam.
            # Create a loop to read the latest frame from the camera using VideoCapture#read()
            # Convert the frame received from OpenCV to a MediaPipe’s Image object.
            ret, frame = vid.read()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            # Send live image data to perform gesture recognition.
            # The results are accessible via the `result_callback` provided in
            # the `GestureRecognizerOptions` object.
            # The gesture recognizer must be created with the live stream mode.
            frame_timestamp_ms = int(vid.get(cv2.CAP_PROP_POS_MSEC))
            recognizer.recognize_async(mp_image, frame_timestamp_ms)

            # Display the resulting frame
            # cv2.imshow('frame', frame)

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            # frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame)

            # Draw the hand annotations on the image.
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            font = cv2.FONT_HERSHEY_SIMPLEX

            # Use putText() method for
            # inserting text on video
            cv2.putText(frame,
                        class_name,
                        (50, 50),
                        font, 1,
                        (0, 255, 255),
                        2,
                        cv2.LINE_4)

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', frame)

            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
