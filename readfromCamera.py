import cv2
import numpy as np
from keras.models import model_from_json

card_dictionary = {
    0: "0 blue", 1: "1 blue", 2: "2 blue", 3: "3 blue", 4: "4 blue", 5: "5 blue",
    6: "6 blue", 7: "7 blue", 8: "8 blue", 9: "9 blue", 10: "reverse blue", 11: "stop blue",
    12: "swipe 2 blue", 13: "0 green", 14: "1 green", 15: "2 green", 16: "3 green", 17: "4 green",
    18: "5 green", 19: "6 green", 20: "7 green", 21: "8 green", 22: "9 green", 23: "reverse green",
    24: "stop green", 25: "swipe 2 green", 26: "0 red", 27: "1 red", 28: "2 red", 29: "3 red",
    30: "4 red", 31: "5 red", 32: "6 red", 33: "7 red", 34: "8 red", 35: "9 red", 36: "reverse red",
    37: "stop red", 38: "swipe 2 red", 39: "0 yellow", 40: "1 yellow", 41: "2 yellow", 42: "3 yellow",
    43: "4 yellow", 44: "5 yellow", 45: "6 yellow", 46: "7 yellow", 47: "8 yellow", 48: "9 yellow",
    49: "reverse yellow", 50: "stop yellow", 51: "swipe 2 yellow"
}

def crop_cards(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use a simple threshold to segment the cards from the background
    _, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours to find bounding boxes around cards
    card_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        card_boxes.append((x, y, x+w, y+h))

    # Return the bounding boxes of detected cards
    return card_boxes



json_file = open('UNO_CNN_Model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("UNO_CNN_Model_complete.h5")
print("Loaded model from disk")

# pass here your video path
cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    # Crop cards from the frame
    card_boxes = crop_cards(frame)
    # Perform any required preprocessing on the frame before prediction
    for box in card_boxes:
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Card Detection", frame)

    # Assuming your model expects a certain input shape, resize and normalize the image
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0  # Normalize pixel values to be between 0 and 1
    input_data = np.expand_dims(normalized_frame, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = emotion_model.predict(input_data)
    predicted_class = np.argmax(prediction)
    
    # Get the corresponding card label from the dictionary
    card_label = card_dictionary[predicted_class]
    
    # Display the card label on the frame
    cv2.putText(frame, str(card_label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow("Card Prediction", frame)
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

