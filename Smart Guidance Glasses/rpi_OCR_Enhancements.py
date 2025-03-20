import socket
import time
import os, sys
import base64
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import tflite_runtime.interpreter as tflite
from PIL import Image

# Initialize camera
camera = PiCamera()
camera.resolution = (1024, 768)
rawCapture = PiRGBArray(camera)

# Set up TFLite interpreter for OCR model
def load_ocr_model():
    # Load TFLite model
    interpreter = tflite.Interpreter(model_path="/home/pi/ProjectV2/ocr_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

# Process image for OCR
def preprocess_image_for_ocr(image_path):
    img = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    # Find text regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find potential text regions
    text_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 15 and h > 15:  # Filter small noise
            text_regions.append((x, y, w, h))
    
    return img, text_regions

# Perform OCR using TinyML
def perform_ocr(interpreter, img, regions):
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Expected input shape for the model
    input_shape = input_details[0]['shape']
    height = input_shape[1]
    width = input_shape[2]
    
    extracted_text = []
    
    for (x, y, w, h) in regions:
        roi = img[y:y+h, x:x+w]
        # Resize ROI to match model input size
        roi_resized = cv2.resize(roi, (width, height))
        # Prepare input data
        input_data = np.expand_dims(roi_resized, axis=0).astype(np.float32)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        # Run inference
        interpreter.invoke()
        
        # Get output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        # Process output to get text
        # This depends on your specific model output format
        text = decode_output(output_data)
        extracted_text.append(text)
    
    return " ".join(extracted_text)

# Decode model output to text
def decode_output(output_data):
    # This function will vary based on your specific model
    # For example, if your model outputs character probabilities:
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = ""
    
    # Simple example - in reality, you'd implement CTC decoding or similar
    for timestep in output_data[0]:
        char_idx = np.argmax(timestep)
        if char_idx < len(alphabet):
            text += alphabet[char_idx]
    
    return text

def image_to_str(image_path):
    with open(image_path, "rb") as img_file:
        im_string = base64.b64encode(img_file.read())
    return im_string

def setup_client(host, port):
    client = None
    address = (host, port)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(address)
    return client

def receive_message_via_socket(client):
    message = None
    message = (client.recv(1024)).decode()
    return message

def send_message_via_socket(client, message, byte_flag=False):
    if not byte_flag:
        client.sendall(message.encode())
    else:
        client.sendall(message)

if __name__ == "__main__":
    print('Entered Main')
    host = '172.27.80.1'  # server ip needs to be updated here
    port = 5050
    print('starting client setup')
    try:
        client = setup_client(host, port)
        print('client setup done')
    except socket.error as error:
        print("Error in setting up server")
        print(error)
        sys.exit()
    
    # Load OCR model
    print('Loading OCR model')
    try:
        ocr_interpreter = load_ocr_model()
        print('OCR model loaded successfully')
    except Exception as e:
        print(f"Error loading OCR model: {e}")
        sys.exit()
    
    bt_fail = False
    print('Starting bluealsa')
    try:
        os.system("sudo service bluealsa start")
    except:
        bt_fail = True
        print("Could not start bluealsa")
    
    if not bt_fail:
        print("bluealsa started")
    
    message = "Hello From Client"
    send_message_via_socket(client, message)
    send_message_via_socket(client, '@end@')
    print("__>> Message Sent")
    time.sleep(1)
    message_recv = receive_message_via_socket(client)
    print("__>> Message Received : [ "+message_recv+" ]")
    
    while True:
        try:
            # Capture image
            camera.capture("/home/pi/ProjectV2/img.png")
            print("Image captured")
            time.sleep(1)
            
            # Perform OCR on the image
            img, text_regions = preprocess_image_for_ocr("/home/pi/ProjectV2/img.png")
            ocr_text = perform_ocr(ocr_interpreter, img, text_regions)
            print(f"OCR Result: {ocr_text}")
            
            # Send both image and OCR text to server
            send_message_via_socket(client, image_to_str('/home/pi/ProjectV2/img.png'), byte_flag=True)
            send_message_via_socket(client, '@end@')
            print("__>> Image string Sent")
            
            # Send OCR text
            send_message_via_socket(client, f"OCR:{ocr_text}")
            send_message_via_socket(client, '@end@')
            print("__>> OCR text Sent")
            
            message_recv = receive_message_via_socket(client)
            print("__>> Message Received : [ "+message_recv+" ]")
            
            person = receive_message_via_socket(client)
            print("__>> Person message Received :")
            
            if person != "Not Found":
                print('Hello '+person+' !')
                try:
                    os.system(f"espeak 'Hello {person}' -ven+f5 -k5 -s150 --stdout | aplay -D bluealsa:DEV=41:42:E8:BE:E0:B1,PROFILE=a2dp")
                except:
                    print("Audio not connected")
                    
                # If OCR text is present, speak it as well
                if ocr_text:
                    try:
                        os.system(f"espeak 'I see text: {ocr_text}' -ven+f5 -k5 -s150 --stdout | aplay -D bluealsa:DEV=41:42:E8:BE:E0:B1,PROFILE=a2dp")
                    except:
                        print("Audio not connected")
                
                time.sleep(1)
                
        except socket.error as error:
            print("Error in communication with server")
            print(error)
            sys.exit()
        except Exception as error:
            print("Error in Main thread:", error)
            print('Repeating Main capture')
    
    camera.close()
