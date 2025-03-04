import socket
import time
import os, sys
import base64

from picamera.array import PiRGBArray
from picamera import PiCamera

camera = PiCamera()
camera.resolution = (1024, 768)

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

        host = '172.27.80.1' #server ip needs to be updated here
        port = 5050

        print('starting client setup')
        try:
                client = setup_client(host, port)
                print('client setup done')

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
                        print("__>> Person message Received :")
                        if person != "Not Found":
                                print('Hello '+person+' !')
                                try:
                                        os.system("espeak 'Hello {}' -ven+f5 -k5 -s150 --stdout | aplay -D bluealsa:DEV=41:42:E8:BE:E0:B1,PROFILE=a2dp".format(person))
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
