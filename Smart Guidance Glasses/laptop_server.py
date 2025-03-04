import socket
import time
import os, sys
import traceback
import numpy as np
import cv2
import face_recognition
import base64

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

if __name__ == "__main__":
	host = ''
	port = 5050
	
	known_encodings=[]
	known_names=[]
	print('Loading known images')
	for subdir in os.listdir('../Images_train'):
		for imfile in os.listdir('../Images_train/'+subdir):
			known_names.append(subdir)
			curr_img = cv2.cvtColor(cv2.imread('../Images_train/'+subdir+'/'+imfile), cv2.COLOR_BGR2RGB)
			known_encodings.append(face_recognition.face_encodings(curr_img)[0])
	print('Loaded known images')


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

	except (KeyboardInterrupt, socket.error) as error:
		print("Error in setting up connection")
		print(error)
		sys.exit()
	
	message_recv = receive_message_via_socket(connection)
	print("__>> Message Received : [ "+message_recv+" ]")
	
	time.sleep(1)
	send_message_via_socket(connection,"Ack1 from Server")
	print("__>> Ack Sent")
	
	while True:
		try:
			im_str = receive_message_via_socket(connection)
			print("__>> Image String Received ") #: '"+im_str+"'")
			time.sleep(1)
			send_message_via_socket(connection,"Ack2 from Server")
			print("__>> Ack Sent")
		except socket.error as error:
			print("Error in communication with server")
			print(error)
			sys.exit()
		
		clicked = str_to_image(im_str)
		
		gotface = True
		clicked_encoding_list = face_recognition.face_encodings(clicked)
		if len(clicked_encoding_list)==0:
			gotface = False
		else:
			clicked_encoding = clicked_encoding_list[0]
			matches = face_recognition.compare_faces(known_encodings, clicked_encoding)
			print(matches)
			if len(matches)==0:
				gotface = False
			else:
				match_names = [known_names[i] for i in range(len(matches)) if matches[i]==True]
				print(match_names)
				if len(match_names)==0:
					gotface = False
				else:
					recognized_person = max(set(match_names), key = match_names.count)
					print(recognized_person)
					send_message_via_socket(connection,recognized_person)
					print("__>> Person Sent")
		
		if not gotface:
			send_message_via_socket(connection,"Not Found")
			print("__>> 'Not found' Sent")
	
	connection.close()
	server.close()
	print("\nSocket Connections closed !!")
