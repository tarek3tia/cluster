import socket

# Define server address and port
SERVER_HOST = '127.0.0.1'  # Change this to the server's IP address
SERVER_PORT = 12345

# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the server
client_socket.connect((SERVER_HOST, SERVER_PORT))

# Send data to server
client_socket.sendall(b'Hello from client!')

# Receive response from server
response = client_socket.recv(1024)
print('Response from server:', response.decode())

# Close the connection
client_socket.close()
