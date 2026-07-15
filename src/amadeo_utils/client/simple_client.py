
from typing import Dict, Any, Optional, Callable
import logging
import json
import socket
import struct

"""
The idea behind this class is to handle all of the work of a simple client - connect to a host and port, communicate with the server, handle the response, and then close the connection (so its a simple get-request-handle-it-respond-shut down). The linchpin for this is it takes, as a variable,
a 'callback function' (callback) method that can be called during the processing of the server response. As an example, I use this as an F5-TTS service, which accepts a voice identifier and text, and the server responds by changing that text to speech (using the given identified voice). The SimpleClient 
is defined like so - note the handle_server_response, that is a method in that particular python script (but not natively in this script):
```
        self.socket_client = SimpleClient(self.host, self.port, additional_server_response_functionality = self.handle_server_response)
```

'handle_server_response' does everything that is specific to the F5-TTS client. 'handle_server_response' is stored in 'additional_server_response_functionality' as a callback function; later, its called like so:
```
                if self.additional_server_response_functionality is not None:
                    json_dict, raw_data = self.additional_server_response_functionality(response, raw_data)
```

In this example, this method actually performs the results of the text-to-speech model, but...it could do anything, really.

The client and server use JSON for communication, and allow for byte data to be tacked on just after the JSON (.wav, .mp3. movies, pictures, even text files, if you encode in UTF-8 a la `"Hello world!".encode('utf-8')`). There is no structure to the JSON the server accepts from the client;
the server does respond with some limited structure, though, as the client needs to determine if it is just getting back JSON _OR_ if its getting back JSON AND raw media data. The callback function accepts this JSON dictionary plus, potentially, that raw media data, but it will not return anything (as 
nothing is left to do from a simple client, outside of shut down). This dictionary required fields that must be sent to the client:
{
    'success': True/False,   # boolean representing if the call was successful
    'message': '',  # any additional message (or error); does not have to be populated, but it must exist
    'file_size': 0  # Integer representing the number of bytes to expect from the raw data (0 if not used). IF you send data and do not set this to len(raw_data), you probably do not need to
                    worry - its set in this code. That said its probably best practice to set this.
}

This dictionary was converted from binary to JSON, at which point we can interact with it.  

"""


# Configure logging to show timestamps and log levels
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

class SimpleClient:

    def __init__(self, host='localhost', port=8888, additional_server_response_functionality: Optional[Callable[[Dict[str, Any], bytes], None]]  = None):
        """
        Initialize the client

        Args:
            host: IP address if the server (default: localhost)
            port: Port number if the server (default: 8888)
            additional_server_response_functionality: A method that will be called from this class to handle any additional functionality surrounding the server's response that will need to be handled by the client.
                                            For arguments, the method expects a (JSON) dictionary as well as POSSIBLY raw data (.wav. mp3, .mov. .wmv, .png, .jpeg, any media file), but only if its warranted. The callback function
                                            will return nothing (as nothing is needed for a simple client). The dictionary being sent to the server is arbitrary (whatever it needs), but the dictionary returning from the server will,
                                            at the very least, _MUST_ have at least these fields:
                                            {
                                                'success': True/False,   # boolean representing if the call was successful
                                                'message': '',  # any additional message (or error); does not have to be populated, but it must exist
                                                'file_size': 0  # Integer representing the number of bytes to expect from the raw data (0 if not used). IF you send data and do not set this to len(raw_data), you probably do not need to
                                                                worry - its set in this code. That said its probably best practice to set this.
                                            }
        """
        self.host = host
        self.port = port

        if additional_server_response_functionality is not None:
            self.additional_server_response_functionality = additional_server_response_functionality
        else:
            logger.warning(f"No additional_server_response_functionality set; this will be a purposeless client.")



    def send_request_to_server(self, data:bytes) -> None:
        """
        Send request to server and handle response
        Args:
            data:   (bytes) The JSON, and any other media file, clumped together in byte form; this clump will be sent to the server (which will unravel the JSON and then any other media file attached).

        """
        try:
            # Create socket connection
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(30)  # 30 second timeout

            try:
                client_socket.connect((self.host, self.port))

                # Send the request to the server
                logger.info(f"Sending request to server...")
                client_socket.sendall(data)

                ######## At this point, we have received a response from the server ########

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

                # unravel the JSON
                response = json.loads(json_data.decode('utf-8'))

                # Receive raw file
                raw_data = None

                # 'success' is expected in the JSON
                if response['success'] == True:

                    # 'file_size' is expected in the JSON
                    file_size = response['file_size']
                    if file_size > 0: logger.info(f"Receiving raw data file ({file_size} bytes)")
                    raw_data = b''
                    while len(raw_data) < file_size:
                        chunk = client_socket.recv(min(8192, file_size - len(raw_data)))
                        if not chunk:
                            raise ValueError("Connection closed while receiving audio")
                        raw_data += chunk

                # Handle any additional things that must be done - this is the callback function, so the meat of your actual code is processed with this method call
                self.additional_server_response_functionality(response, raw_data)

            finally:
                client_socket.close()

        except Exception as e:
            logger.error(f"Request failed: {e}")
