import socket
import time
import os, sys, select
import base64

from picamera.array import PiRGBArray
from picamera import PiCamera

import RPi.GPIO as GPIO

##############################################################

GPIO.setmode(GPIO.BCM)
BUTTON_GPIO = 22
flag_kb = 1
inp_name = "Demo_Name"

camera = PiCamera()
camera.resolution = (1024, 768)

def drop_flag_kb(chan):
    global flag_kb
    global BUTTON_GPIO
    flag_kb = 0
    print('flag dropped')
    print("Stop Speech to text here")
    # code to switch off mic
    GPIO.remove_event_detect(BUTTON_GPIO)

def raise_flag_kb():
    global flag_kb
    flag_kb = 1
    print('flag raised')

def name_input():
    global flag_kb
    global inp_name
    print('Input Function with flag = ', flag_kb)
    print("Start speech here [Speech to text]")
    while flag_kb==1:
        # turn on bluetooth-4 mic here
        # convert from speech to text and store in inp_name
        time.sleep(0.1)
    inp_name = input("No Speech, Enter Name: -->")
    flag_kb=1
    print('flag raised')

def new_person(client):
    global camera
    global inp_name
    global BUTTON_GPIO

    while True:    
        print("----------- NEW PERSON -------------")
        send_message_via_socket(client,'$NEW_PERSON$@end@')

        # name = input("Provide the persons's name : ")
        GPIO.setup(BUTTON_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(BUTTON_GPIO, GPIO.RISING, callback=drop_flag_kb, bouncetime=200)
        name_input()
        name = inp_name
        send_message_via_socket(client,name)
        send_message_via_socket(client,"@end@")
        print(f"         _>> Name Sent : [ {name} ]")

        for i in range(5):
            camera.capture("/home/pi/ProjectV2/img.png")
            print("Image captured")
            send_message_via_socket(client,image_to_str('/home/pi/ProjectV2/img.png'),byte_flag=True)
            send_message_via_socket(client,'@end@')
            print("         _>> Image "+str(i+1)+" sent")
            time.sleep(1)
            message_recv = receive_message_via_socket(client)

            print(f"         _>> Message Received [ {message_recv} ] ")
            if message_recv == "Error for this image":
                break
        
        option = input("Continue to new person? [Y/N]")
        if option=='Y' or option=='y':
            continue

        send_message_via_socket(client,'$END_Process$@end@')        
        print("-------------- DONE ----------------")
        return

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

        host = '100.68.20.244' #server ip needs to be updated here
        port = 5050
        port2 = 5051

        print('starting client setup')
        try:
                client = setup_client(host, port)
                print('client setup done')
                client = setup_client(host, port2)
                print('client2 setup done')

        except socket.error as error:
                print("Error in setting up server")
                print(error)
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
        send_message_via_socket(client,message)
        send_message_via_socket(client,'@end@')
        print("__>> Message Sent")
        time.sleep(1)
        message_recv = receive_message_via_socket(client)
        print("__>> Message Received : [ "+message_recv+" ]")

        while True:
                try:
                        camera.capture("/home/pi/ProjectV2/img.png")
                        print("Image captured")
                        time.sleep(1)
                        
                        send_message_via_socket(client,image_to_str('/home/pi/ProjectV2/img.png'),byte_flag=True)
                        send_message_via_socket(client,'@end@')
                        print("__>> Image string Sent")
                        message_recv = receive_message_via_socket(client)
                        print("__>> Message Received : [ "+message_recv+" ]")

                        person = receive_message_via_socket(client)
                        print("__>> Person message Received :", person)
                        if person == "New Face":
                            print("Do you want to add the new face? [y/n] --> ", end="")
                            opt_status, dummy1, dummy2 = select.select(sys.stdin.readline(), [], [], 1.5)
                            if not opt_status:
                                opt = 'n'
                            else:
                                opt = sys.stdin.readline().strip()

                            send_message_via_socket(client,opt+'@end@') #2
                            ready_ack = receive_message_via_socket(client)
                            print("__>> Readiness Acknowledgement:", ready_ack)
                            if opt == 'y' or opt=='Y':
                                new_person(client)

                        if person != "Not Found":
                                print('Hello '+person+' !')
                                try:
                                        os.system("espeak 'Hello {}' -ven+f5 -k5 -s150 --stdout | aplay -D bluealsa:DEV=30:21:99:C2:4E:A6,PROFILE=a2dp".format(person))
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
