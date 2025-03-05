import cv2
import mediapipe as mp
import math
import numpy as np

class GestureLightDimmer:
    def __init__(self):
        # MediaPipe Hand Tracking Setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Light dimmer variables
        self.brightness_level = 50  # Initial brightness (0-100)
        self.min_brightness = 0
        self.max_brightness = 100
        
        # Hand tracking setup
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def calculate_hand_spread(self, hand_landmarks):
        """
        Calculate the spread between thumb and pinky to control brightness.
        
        :param hand_landmarks: MediaPipe hand landmarks
        :return: Distance between thumb tip and pinky tip
        """
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        
        # Calculate Euclidean distance
        distance = math.sqrt(
            (thumb_tip.x - pinky_tip.x)**2 + 
            (thumb_tip.y - pinky_tip.y)**2
        )
        return distance

    def map_brightness(self, hand_spread, min_spread=0.1, max_spread=0.5):
        """
        Map hand spread to brightness level.
        
        :param hand_spread: Distance between thumb and pinky
        :param min_spread: Minimum spread for lowest brightness
        :param max_spread: Maximum spread for highest brightness
        :return: Brightness level (0-100)
        """
        # Constrain the spread
        spread = max(min_spread, min(hand_spread, max_spread))
        
        # Map spread to brightness
        brightness = int(
            ((spread - min_spread) / (max_spread - min_spread)) * 100
        )
        return max(0, min(brightness, 100))

    def detect_on_off_gesture(self, hand_landmarks):
        """
        Detect on/off gesture (fist or open hand).
        
        :param hand_landmarks: MediaPipe hand landmarks
        :return: True if on/off gesture detected
        """
        # Get finger tip landmarks
        landmarks = hand_landmarks.landmark
        
        # Check for closed fist (all fingers close to palm)
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        fingertips = [
            landmarks[self.mp_hands.HandLandmark.THUMB_TIP],
            landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
            landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
            landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP],
            landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
        ]
        
        # Check if all fingertips are close to wrist (fist)
        is_fist = all(
            math.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2) < 0.1 
            for tip in fingertips
        )
        
        return is_fist

    def run(self):
        """
        Main method to run gesture-controlled light dimmer.
        """
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        # Light state
        light_on = True
        
        while cap.isOpened():
            # Read frame from webcam
            success, frame = cap.read()
            if not success:
                break
            
            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame and detect hands
            results = self.hands.process(rgb_frame)
            
            # Reset brightness display
            display_text = f"Brightness: {self.brightness_level}%"
            
            # Draw initial text
            cv2.putText(frame, display_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Check if hands are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Check for on/off gesture
                    if self.detect_on_off_gesture(hand_landmarks):
                        light_on = not light_on
                        cv2.putText(frame, 
                                    f"Light {'On' if light_on else 'Off'}", 
                                    (10, 70), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                    (0, 255, 0), 2)
                    
                    # If light is on, adjust brightness
                    if light_on:
                        # Calculate hand spread
                        hand_spread = self.calculate_hand_spread(hand_landmarks)
                        
                        # Map hand spread to brightness
                        self.brightness_level = self.map_brightness(hand_spread)
                        
                        # Update brightness display
                        display_text = f"Brightness: {self.brightness_level}%"
                        cv2.putText(frame, display_text, (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                    (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Gesture Light Dimmer', frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

# Run the gesture light dimmer
if __name__ == "__main__":
    dimmer = GestureLightDimmer()
    dimmer.run()
