
from typing import Dict, Any, Optional, Callable
import logging
import json
import socket
import struct

"""
This is the enhanced 'simple client'. Its enhanced over the basic simple_client due to the following:
* The connection can be persistent or transient; if persistent,
  * This requires a sessionID  
    * The only communication that will not have the sessionID will be the initial connection from the client to the server  
    * In the initial exchange, the server will come up with the sessionID and send that back in its first response, and the client will record this sessionID and put that in every message moving forward. 
* Both the client AND the server can send AND receive arbitrary binary files.  

The idea behind this class is to handle all of the work of a simple client - connect to a host and port, communicate with the server, handle the response, and then either close the connection (so its a simple get-request-handle-it-respond-shut down) OR treat it like a permanent connection. The linchpin for 
this is it takes, as a variable, a 'callback function' (callback) method that can be called during the processing of the server response. As an example, I use this as an F5-TTS service, which accepts a voice identifier and text, and the server responds by changing that text to speech (using the given identified 
voice). The SimpleClient is defined like so - note the handle_server_response, that is a method in that particular python script (but not natively in this script):
```
        self.socket_client = SimpleClient(self.host, self.port, additional_server_response_functionality = self.handle_server_response)
```

'handle_server_response' does everything that is specific to the F5-TTS client. 'handle_server_response' is stored in 'additional_server_response_functionality' as a callback function; later, its called like so:
```
                if self.additional_server_response_functionality is not None:
                    json_dict, raw_data = self.additional_server_response_functionality(response, raw_data)
```

In this example, this method actually performs the results of the text-to-speech model, but...it could do anything, really.

The client and server use JSON for communication, and allow for byte data to be tacked on just after the JSON (.wav, .mp3. movies, pictures, even text files, if you encode in UTF-8 a la `"Hello world!".encode('utf-8')`). The structure of the JSON that the client sends to the server contains the following:
{
    'persistent': bool, # if this connection should last until the client sends a terminate message, this is true; if this is a on-and-done' request / response, this should be false
    'sessionID': str,   # The session ID; will be blank for the initial connection / request to the server, but will be populated in every message after that. If the client sends a non empty string sessionID to the server in a non-persistent connection, that sessionID will be returned. Likewise, 
                    # If the connection is persistent, just being established, and the client sends a sessionID, as long as that sessionID is not in use the server will use it
    'requestID': str,   # A secondary, and child, ID of session ID; you dont have to use this / this can be empty, but sometimes this is helpful for sub-requests.
    'command': str, # What the client is requesting the server to do; the first will always be 'establish_connection', but after that the command is arbitrary / depends on the implementation 
    'message': str,  # any additional message (or error); does not have to be populated, but it must exist
    'file_size': int  # Integer representing the number of bytes to expect from the raw data (0 if not used). IF you send data and do not set this to len(raw_data), you probably do not need to 
                    worry - its set in this code. That said its probably best practice to set this.
}

The dictionary / JSON sent to the client from the server has a few required fields (in addition to additional arbitrary fields), which are similar - but not the same - as the JSON to the server:
{
    'success': True/False,   # boolean representing if the call was successful
    'message': str,  # any additional message (or error); does not have to be populated, but it must exist
    'sessionID': str,   # The session ID;
    'requestID': str,   # A secondary, and child, ID of session ID; you dont have to use this / this can be blank, but sometimes this is helpful for sub-requests.
    'client_address': str, # The IP of the client (as determined by the server)
    'client_port': str, # The port that the client is connected to (as determined by the server)
    'file_size': 0  # Integer representing the number of bytes to expect from the raw data (0 if not used). IF you send data and do not set this to len(raw_data), you probably do not need to 
                    worry - its set in this code. That said its probably best practice to set this.
} 

This dictionary was converted from binary to JSON, at which point we can interact with it.  


Usage Examples:
* Persistent connection:
```
client = EnhancedSimpleClient(host, port, callback)
if client.establish_persistent_connection():
    client.send_persistent_request("process_audio", "Convert this", audio_data)
    client.send_persistent_request("another_command", "More work")
    client.close_connection()
```

* Transient connection:
```
client = EnhancedSimpleClient(host, port, callback)  
response, data = client.send_transient_request("quick_task", "One-time request", binary_data)
```
"""

# Configure logging to show timestamps and log levels
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

class AmadeoClient:

    PERSISTENT_RESPONSE_TIMEOUT = 30

    def __init__(self, host='localhost', port=8888, additional_server_response_functionality: Optional[Callable[[Dict[str, Any], Optional[bytes]], None]]  = None, session_id: str = '', request_id: str = '', persistent_request_timeout: int = PERSISTENT_RESPONSE_TIMEOUT):
        """
        Initialize the client

        Args:
            host: IP address if the server (default: localhost)
            port: Port number if the server (default: 8888)
            additional_server_response_functionality: A method that will be called from this class to handle any additional functionality surrounding the server's response that will need to be handled by the client.
                                            For arguments, the method expects a (JSON) dictionary as well as POSSIBLY raw data (.wav. mp3, .mov. .wmv, .png, .jpeg, any media file), but only if its warranted. The callback function
                                            will return nothing (as nothing is needed for a simple client). Refer to the JSON descriptions above for what the client needs to send to the server / the basics of what it should expect in return
            session_id: if you wish to _attempt_ to establish the sessionID you may do so. This is helpful if you want to use the same sessionID throughout several different services. Be warned, though, that the server may not be able to accommodate the request if another session has that ID
            persistent_request_timeout: The timeout, in seconds, of a persistent request.
        """
        self.host = host
        self.port = port
        self.client_socket = None
        self.is_persistent = False
        self.session_id = session_id
        self.request_id = request_id
        self.persistent_request_timeout = persistent_request_timeout

        if additional_server_response_functionality is not None:
            self.additional_server_response_functionality = additional_server_response_functionality
        else:
            self.additional_server_response_functionality = None
            logger.warning(f"No additional_server_response_functionality set; this will be a purposeless client.")

    def establish_persistent_connection(self):
        """
        Establish a persistent connection with the server

        Returns:
            bool: True if connection established successfully, False otherwise
        """
        try:
            # Create socket connection
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.settimeout(self.persistent_request_timeout)
            self.client_socket.connect((self.host, self.port))

            # Send establish_connection request
            request_data = {
                'persistent': True,
                'sessionID': self.session_id,
                'requestID': self.request_id,
                'command': 'establish_connection',
                'message': 'Requesting persistent connection',
                'file_size': 0
            }

            # Send the request
            self.send_request_data(request_data, None)

            # Receive response
            response, raw_data = self.receive_response()

            if response and response.get('success'):
                self.session_id = response.get('sessionID')
                self.is_persistent = True
                logger.info(f"Persistent connection established with session ID: {self.session_id}")
                return True
            else:
                logger.error(f"Failed to establish persistent connection: {response.get('message', 'Unknown error')}")
                self.close_connection()
                return False

        except Exception as e:
            logger.error(f"Failed to establish persistent connection: {e}")
            self.close_connection()
            return False

    def close_connection(self):
        """
        Close the connection (persistent or transient)
        """
        if self.client_socket:
            try:
                if self.is_persistent and self.session_id:
                    # Send termination request for persistent connections
                    request_data = {
                        'persistent': True,
                        'sessionID': self.session_id,
                        'requestID': self.request_id,
                        'command': 'terminate_session',
                        'message': 'Terminating session',
                        'file_size': 0
                    }
                    self.send_request_data(request_data, None)
                    # Don't wait for response - just close

                self.client_socket.close()
                logger.info(f"Connection closed (Session: {self.session_id})")
            except:
                pass
            finally:
                self.client_socket = None
                self.session_id = None
                self.is_persistent = False

    def send_request_data(self, request_dict, binary_data=None):
        """
        Send request data to server (JSON + optional binary data)

        Args:
            request_dict: Dictionary to be sent as JSON
            binary_data: Optional binary data to send after JSON
        """
        # Ensure file_size is set correctly
        if binary_data:
            request_dict['file_size'] = len(binary_data)
        else:
            request_dict['file_size'] = 0

        # Convert to JSON and encode
        json_data = json.dumps(request_dict).encode('utf-8')

        # Send JSON header (4 bytes) + JSON data
        header = struct.pack('!I', len(json_data))
        self.client_socket.sendall(header + json_data)

        # Send binary data if present
        if binary_data:
            self.client_socket.sendall(binary_data)
            logger.debug(f"Sent {len(binary_data)} bytes of binary data")

    def receive_response(self):
        """
        Receive response from server (JSON + optional binary data)

        Returns:
            tuple: (response_dict, binary_data) where binary_data can be None
        """
        # Receive response header
        header_data = self.client_socket.recv(4)
        if len(header_data) != 4:
            raise ValueError("Invalid response header")

        json_length = struct.unpack('!I', header_data)[0]

        # Receive JSON response
        json_data = b''
        while len(json_data) < json_length:
            chunk = self.client_socket.recv(json_length - len(json_data))
            if not chunk:
                raise ValueError("Connection closed prematurely")
            json_data += chunk

        # Parse JSON
        response = json.loads(json_data.decode('utf-8'))

        # Receive binary data if present
        raw_data = None
        file_size = response.get('file_size', 0)
        if file_size > 0:
            logger.info(f"Receiving raw data file ({file_size} bytes)")
            raw_data = b''
            while len(raw_data) < file_size:
                chunk = self.client_socket.recv(min(8192, file_size - len(raw_data)))
                if not chunk:
                    raise ValueError("Connection closed while receiving binary data")
                raw_data += chunk

        return response, raw_data

    def send_persistent_request(self, command, message='', binary_data=None, **kwargs):
        """
        Send a request on an established persistent connection

        Args:
            command: The command to send to the server
            message: Optional message
            binary_data: Optional binary data to send
            **kwargs: Additional fields to include in the request

        Returns:
            tuple: (response_dict, binary_data) or (None, None) if failed
        """
        if not self.is_persistent or not self.session_id or not self.client_socket:
            logger.error("No persistent connection established")
            return None, None

        try:
            # Prepare request
            request_data = {
                'persistent': True,
                'sessionID': self.session_id,
                'requestID': self.request_id,
                'command': command,
                'message': message,
                'file_size': 0
            }
            # Add any additional fields
            request_data.update(kwargs)

            # Send request
            self.send_request_data(request_data, binary_data)

            # Receive response
            response, raw_data = self.receive_response()

            # Handle response through callback
            if self.additional_server_response_functionality:
                self.additional_server_response_functionality(response, raw_data)

            return response, raw_data

        except Exception as e:
            logger.error(f"Persistent request failed: {e}")
            return None, None

    def send_transient_request(self, command, message='', binary_data=None, **kwargs):
        """
        Send a single transient request (connect, send, receive, disconnect)

        Args:
            command: The command to send to the server
            message: Optional message
            binary_data: Optional binary data to send
            **kwargs: Additional fields to include in the request

        Returns:
            tuple: (response_dict, binary_data) or (None, None) if failed
        """
        client_socket = None
        try:
            # Create socket connection
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(30)
            client_socket.connect((self.host, self.port))

            # Prepare request
            request_data = {
                'persistent': False,
                'sessionID': self.session_id,
                'requestID': self.request_id,
                'command': command,
                'message': message,
                'file_size': 0
            }
            # Add any additional fields
            request_data.update(kwargs)

            # Ensure file_size is set correctly
            if binary_data:
                request_data['file_size'] = len(binary_data)

            # Send request
            json_data = json.dumps(request_data).encode('utf-8')
            header = struct.pack('!I', len(json_data))
            client_socket.sendall(header + json_data)

            if binary_data:
                client_socket.sendall(binary_data)
                logger.debug(f"Sent {len(binary_data)} bytes of binary data")

            # Receive response
            header_data = client_socket.recv(4)
            if len(header_data) != 4:
                raise ValueError("Invalid response header")

            json_length = struct.unpack('!I', header_data)[0]

            # Receive JSON response
            json_data = b''
            while len(json_data) < json_length:
                chunk = client_socket.recv(json_length - len(json_data))
                if not chunk:
                    raise ValueError("Connection closed prematurely")
                json_data += chunk

            response = json.loads(json_data.decode('utf-8'))

            # Receive binary data if present
            raw_data = None
            file_size = response.get('file_size', 0)
            if file_size > 0:
                logger.info(f"Receiving raw data file ({file_size} bytes)")
                raw_data = b''
                while len(raw_data) < file_size:
                    chunk = client_socket.recv(min(8192, file_size - len(raw_data)))
                    if not chunk:
                        raise ValueError("Connection closed while receiving binary data")
                    raw_data += chunk

            # Handle response through callback
            if self.additional_server_response_functionality:
                self.additional_server_response_functionality(response, raw_data)

            return response, raw_data

        except Exception as e:
            logger.error(f"Transient request failed: {e}")
            return None, None
        finally:
            if client_socket:
                try:
                    client_socket.close()
                except:
                    pass

    def send_request_to_server(self, data: bytes) -> None:
        """
        Deprecated!!!

        Legacy method for backward compatibility

        Args:
            data: (bytes) The JSON, and any other media file, clumped together in byte form
        """
        logger.warning("send_request_to_server is deprecated. Use send_transient_request or send_persistent_request instead.")

        try:
            # Create socket connection
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(30)

            try:
                client_socket.connect((self.host, self.port))

                # Send the request to the server
                logger.info(f"Sending request to server...")
                client_socket.sendall(data)

                # Receive response header
                header_data = client_socket.recv(4)
                if len(header_data) != 4:
                    raise ValueError("Invalid response header")

                json_length = struct.unpack('!I', header_data)[0]

                # Receive JSON response
                json_data = b''
                while len(json_data) < json_length:
                    chunk = client_socket.recv(json_length - len(json_data))
                    if not chunk:
                        raise ValueError("Connection closed prematurely")
                    json_data += chunk

                response = json.loads(json_data.decode('utf-8'))

                # Receive raw file
                raw_data = None
                if response.get('success'):
                    file_size = response.get('file_size', 0)
                    if file_size > 0:
                        logger.info(f"Receiving raw data file ({file_size} bytes)")
                        raw_data = b''
                        while len(raw_data) < file_size:
                            chunk = client_socket.recv(min(8192, file_size - len(raw_data)))
                            if not chunk:
                                raise ValueError("Connection closed while receiving audio")
                            raw_data += chunk

                # Handle response through callback
                if self.additional_server_response_functionality:
                    self.additional_server_response_functionality(response, raw_data)

            finally:
                client_socket.close()

        except Exception as e:
            logger.error(f"Request failed: {e}")

    def update_request_id(self, request_id):
        self.request_id = request_id

