import socket
import time
import os, sys
import traceback
import numpy as np
import cv2
import face_recognition
import base64

def add_person(connection):
	
	while True:
		message_recv = receive_message_via_socket(connection)
		print(f"   _>> Message Received : [ {message_recv} ]")
		if(message_recv == "$END_Process$"):
			print("Ending Process from client")
			return
		if(message_recv != "$NEW_PERSON$"):
			continue

		name = receive_message_via_socket(connection)
		print(f"   _>> Name Received : [ {name} ]")
		os.mkdir("../Images_train/"+name)
		for i in range(5):
			try:
				im_str = receive_message_via_socket(connection)
				print("	 _>> Image String Received ") #: '"+im_str+"'")
				
				#save image
				img = str_to_image(im_str)
				img_encoding_list = face_recognition.face_encodings(img)
				if(len(img_encoding_list)==0):
					time.sleep(1)
					send_message_via_socket(connection,"No face found")
					print("	 _>> Ack Sent")
				else:
					cv2.imwrite("../Images_train/"+name+"/"+name+str(i+1)+".jpg" , img)
					time.sleep(1)
					send_message_via_socket(connection,"ImgAck from Server")
					print("	 _>> Ack Sent")

			except socket.error as error:
				print("Error in communication")
				print(error)
				send_message_via_socket(connection,f"Error for this image")
				print("	 _>> Error Sent")

def str_to_image(im_str):
	fh = open('retrieved.png', 'wb') # fh = file handle
	fh.write(base64.b64decode(im_str))
	fh.close()
	return cv2.imread('retrieved.png')

def setup_server(host, port):
	server = None
	server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	address = (host,port)
	server.bind(address)
	print('Server is set up. Host = '+host+' Port = '+str(port))
	print(socket.getaddrinfo(host, port, family=0, type=0, proto=0, flags=0))
	return server

def setup_connection(server):
	connection = None
	address = None
	server.listen()
	print('Listening for requests')
	connection, address = server.accept()
	print('Accepted request')
	print('Connection is set up.')
	return connection, address

def receive_message_via_socket(connection):
	message = ''
	curr = 'dummy'
	while curr != '@end@' or curr != '' or curr!= b'' or curr != None:
		curr = (connection.recv(1024)).decode()
		#print(curr)
		message += curr
		if message.strip()[-5:] == '@end@':
			break
	return message.strip()[:-5]

def send_message_via_socket(connection, message):
	connection.sendall(message.encode())

def flush(connection):
	message = ''
	curr = 'dummy'
	while curr != '@xtr@' or curr != '' or curr!= b'' or curr != None:
		curr = (connection.recv(1024)).decode()
		#print(curr)
		message += curr
		if message.strip()[-5:] == '@xtr@':
			break
	connection.sendall('@xtr@'.encode())
	print("Server Buffer empty, good to go")

if __name__ == "__main__":
	host = ''
	port = 5051

	try:
		server = setup_server(host, port)
		print('server set up done')

	except socket.error as error:
		print("Error in setting up server")
		print(error)
		sys.exit()

	try:
		connection, address = setup_connection(server)
		print("Connected to: " + address[0] + ":" + str(address[1]))
		# flush(connection)

	except (KeyboardInterrupt, socket.error) as error:
		print("Error in setting up connection")
		print(error)
		sys.exit()
	
	add_person(connection)
