from typing import Dict, Any, Optional, Callable, Tuple
import signal
import logging
import json
import socket
import struct
import uuid
import threading
import time

"""
This is the enhanced 'simple server'. Its enhanced over the basic simple_server due to the following:
* The connection can be persistent or transient; if persistent, 
  * This requires a sessionID  
    * The only communication that will not have the sessionID will be the initial connection from the client to the server  
    * In the initial exchange, the server will come up with the sessionID and send that back in its first response, and the client will record this sessionID and put that in every message moving forward. 
* Both the client AND the server can send AND receive arbitrary binary files.  

Each handle_client() is launched in its own thread - so this can, in theory, handle multiple connections at once. That said, if this uses other resources - say, a GPU - you will have to do additional locks / threading in the code that builds off this code. 
 
The idea behind this class is to handle all of the work of a simple server - bind to a host and port, accept client requests (which sometimes have a binary file attached at the end of it), respond (sometimes attaching a binary file at the end of the message), and keeping the connection open 
until the client ends the session.. The linchpin for this is it takes, as a variable, a 'callback function' (callback) method that can be called during a client request. As an example (although the code should be generalized to process ANY type of binary file or request), I use this in an 
F5-TTS service, which accepts a voice identifier and text, and the server responds by changing that to text to speech (using the given identified voice). The f5 object is defined, and then this class is quickly defined like so:  
```
        f5_model = f5.AmadeoF5(voice_samples_dir = voice_samples_dir)
        self.server = EnhancedSimpleServer(host, port, additional_client_functionality = f5_model.handle_client_request)
```

As you can see, the f5_model's 'handle_client_request' is stored in 'additional_client_functionality' as a callback function; later, its called in this class' 'handle_client' like so:
```
                if self.additional_client_functionality is not None:
                    json_dict, raw_data = self.additional_client_functionality(request)
```

In this example, this method actually performs the text-to-speech work, but...it could do anything, really.

It should also be noted that there is a 'synchronous' setting. Basically, this is true when a client request comes in, a single, continuous, synchronous chain of function calls will happen, and then the result is returned to the client.
This would be false if, say, the server got the request, then had to run it through a different service (for example, the client sends audio, and the server sends it to ASR, gets the response from that and sends it to a LLM). If the result 
is not immediately available after running additional_client_functionality this will be False. This is important as otherwise, AmadeoServer will try to send a response to the client, which it cannot do if the result is not immediately available 
from additional_client_functionality.
 
The client and server use JSON for communication, and allow for byte data to be tacked on just after the JSON (.wav, .mp3. movies, pictures, even text files, if you encode in UTF-8 a la `"Hello world!".encode('utf-8')`). The structure of the JSON that the client sends to the server contains the following:
{
    'persistent': bool, # if this connection should last until the client sends a terminate message, this is true; if this is a on-and-done' request / response, this should be false
    'type': str, # Identifies the type. The two base ones for the server are 'connection' and 'error', but there can be additional ones based on the implemented server.
    'sessionID': str,   # The session ID; will be blank for the initial connection / request to the server, but will be populated in every message after that. If the client sends a non empty string sessionID to the server in a non-persistent connection, that sessionID will be returned. Likewise, 
                    # If the connection is persistent, just being established, and the client sends a sessionID, as long as that sessionID is not in use the server will use it
    'requestID': str,   # A secondary, and child, ID of session ID; you dont have to use this / this can be empty, but sometimes this is helpful for sub-requests. 
    'command': str, # What the client is requesting the server to do; the first will always be 'establish_connection', but after that the command is arbitrary / depends on the implementation 
    'message': str,  # any additional message (or error); does not have to be populated, but it must exist
    'file_size': int  # Integer representing the number of bytes to expect from the raw data (0 if not used). IF you send data and do not set this to len(raw_data), you probably do not need to 
                    worry - its set in this code. That said its probably best practice to set this.
}

This JSON, converted to a dictionary, will be sent to the callback function for further processing. The callback function MUST return a tuple (dictionary, binary), where the 'binary' is the raw data (but this can also be None, if the communication only contains the JSON).  
  
The dictionary / JSON sent to the client has a few required fields, which are similar - but not the same - as the JSON to the server:
{
    'success': True/False,   # boolean representing if the call was successful
    'message': str,  # any additional message (or error); does not have to be populated, but it must exist
    'sessionID': str,   # The session ID;
    'requestID': str,   # A secondary, and child, ID of session ID; you dont have to use this / this can be blank, but sometimes this is helpful for sub-requests.
    'client_address': str, # The IP of the client (as determined by the server)
    'client_port': str, # The port that the client is connected to (as determined by the server)
    '_client_info': Dict, # A dictionary structured as {'socket': client_socket, 'address': address}. This stores the actual Client_socket OBJECT and the address (usually a host:port tuple). This is moreso for internal server handling of the 
                            # message, as in some cases, the return call is not straightforward / cant be handled In a synchronous fashion; this request may be sent to other processes and we will lose the chain of custody / command 
    'file_size': 0  # Integer representing the number of bytes to expect from the raw data (0 if not used). IF you send data and do not set this to len(raw_data), you probably do not need to 
                    worry - its set in this code. That said its probably best practice to set this.
} 

This dictionary is converted to JSON, converted to binary, and then the byte data from any media file is tacked onto the end (if it exists), and then the data is sent to the client.

The server will run indefinitely, servicing clients, until Ctrl-C is pressed, a SIGTERM is sent, etc to stop it (or it is otherwise killed).   
"""

# Configure logging to show timestamps and log levels
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

class AmadeoServer:

    CLIENT_TIMEOUT = 300 # timeout for client, in seconds. 300 = 5 minutes

    additional_shutdown: Optional[Callable[[str], None]]

    def __init__(self, host='localhost', port=8888, synchronous=True, client_timeout = CLIENT_TIMEOUT, additional_client_functionality: Optional[Callable[[Dict[str, Any], Optional[bytes]], Tuple[Dict[str, Any], Optional[bytes]]]]  = None, additional_shutdown: Optional[Callable[[str], None]] = None):
        """
        Initialize the server

        Args:
            host: IP address to bind to (default: localhost)
            port: Port number to listen on (default: 8888)
            synchronous: if the call is synchronous. Basically, this is true when a client request comes in, a single, continuous, synchronous chain of function calls will happen, and then the result is returned to the client.
                         This would be false if, say, the server got the request, then had to run it through a different service (for example, the client sends audio, and the server sends it to ASR, gets the response from that and
                         sends it to a LLM). If the result is not immediately available after running additional_client_functionality this will be False. This is important as otherwise, AmadeoServer will try to send a
                         response to the client, which it cannot do if the result is not immediately available from additional_client_functionality.
            additional_client_functionality: A method that will be called from this class to handle any additional functionality surrounding a client request that will need to be handled by this server.
                                            The method expects a dictionary and optional binary data as its args, and should return a dictionary (that will be turned into JSON) and byte data (if applicable)
                                            The dictionary being sent to the method is arbitrary (whatever it needs); the dictionary returned by the method _MUST_ have at least these fields:
                                            {
                                                'success': True/False,   # boolean representing if the call was successful
                                                'message': '',  # any additional message (or error); does not have to be populated, but it must exist
                                                'file_size': 0  # Integer representing the number of bytes to expect from the raw data (0 if not used). IF you send data and do not set this to len(raw_data), you probably do not need to
                                                                worry - its set in this code. That said its probably best practice to set this.
                                            }
            additional_shutdown: A method that will be called from this class during shutdown. The single parameter is a string and will usually represent the session_id; this is helpful if
                                the additional functionality has state that must be cleared upon exit. This is completely optional.
        """
        self.host = host
        self.port = port
        self.client_timeout = client_timeout
        self.synchronous = synchronous

        # Flag to control server shutdown
        # This is set by the signal handler when Ctrl-C is pressed
        self.shutdown_requested = False

        # Dictionary to track active sessions
        # Key: sessionID, Value: dict with client info and socket
        self.active_sessions = {}
        self.sessions_lock = threading.Lock() #called in a while a la 'with self.sessions_lock:', which waits till the lock is free, then claims the lock for itself, updates, and releases the lock

        if additional_client_functionality is not None:
            self.additional_client_functionality = additional_client_functionality
        else:
            self.additional_client_functionality = None
            logger.warning(f"No additional_client_functionality set; this will be a purposeless server.")

        if additional_shutdown is not None:
            self.additional_shutdown = additional_shutdown
        else:
            self.additional_shutdown = None


    def signal_handler(self, signum, frame):
        """
        Handle shutdown signals (Ctrl-C, SIGTERM, etc.)

        This is called when the user presses Ctrl-C or the process receives a termination signal.
        Instead of immediately killing the process, we set a flag that will cause the main
        server loop to exit gracefully after the current operation completes.

        Args:
            signum: Signal number (e.g., SIGINT for Ctrl-C)
            frame: Current stack frame (not used but required by signal handler interface)
        """
        signal_names = {signal.SIGINT: "SIGINT (Ctrl-C)", signal.SIGTERM: "SIGTERM"}
        signal_name = signal_names.get(signum, f"Signal {signum}")
        logger.info(f"Received {signal_name}, requesting graceful shutdown...")

        # Set flag to stop the main server loop
        # The server will finish handling any current client and then shut down
        self.shutdown_requested = True

    def reset_additional_client_functionality(self, additional_client_functionality: Optional[Callable[[Dict[str, Any], Optional[bytes]], Tuple[Dict[str, Any], Optional[bytes]]]]):
        """
        Re-sets additional_client_functionality

        Args:
            additional_client_functionality: A method that will be called from this class to handle any additional functionality surrounding a client request that will need to be handled by this server.
                                            The method expects a dictionary and optional binary data as its args, and should return a dictionary (that will be turned into JSON) and byte data (if applicable)
                                            The dictionary being sent to the method is arbitrary (whatever it needs); the dictionary returned by the method _MUST_ have at least these fields:
                                            {
                                                'success': True/False,   # boolean representing if the call was successful
                                                'message': '',  # any additional message (or error); does not have to be populated, but it must exist
                                                'file_size': 0  # Integer representing the number of bytes to expect from the raw data (0 if not used). IF you send data and do not set this to len(raw_data), you probably do not need to
                                                                worry - its set in this code. That said its probably best practice to set this.
                                            }
        """
        self.additional_client_functionality = additional_client_functionality

    @staticmethod
    def generate_session_id():
        """
        Generate a unique session ID using UUID4

        Returns:
            str: A unique session identifier
        """
        return str(uuid.uuid4())

    def send_response(self, client_socket, response_json, raw_data = None):
        """
        Send response back to client

        This function implements the response protocol:
        1. Send 4-byte header with JSON length
        2. Send JSON with success/metadata
        3. If raw data is available, it is sent as well

        The client expects this exact format, so any changes here must be
        coordinated with changes in the client code.

        The dictionary representing the JSON sent back to the client _MUST_ have at least these fields:
        {
            'success': True/False,   # boolean representing if the call was successful
            'message': '',  # any additional message (or error); does not have to be populated, but it must exist
            'file_size': 0  # Integer representing the number of bytes to expect from the raw data (0 if not used). IF you send data and do not set this to len(raw_data), you probably do not need to
                            worry - its set in this code. That said its probably best practice to set this.
        }

        Args:
            client_socket: Connected client socket
            response_json: a dictionary representing the JSON string you wish to send to the client
            raw_data: the raw data to be sent to the client (if necessary). This is agnostic enough to send wav files, mp3s, movie files, pictures, and text files as raw_data, but if you are sending text you must use something like `"Hello world!".encode('utf-8')`

        Raises:
            Exception
        """
        try:

            logger.debug(f"Sending response..:")

            if not response_json.get('file_size'):
                if raw_data: response_json['file_size'] = len(raw_data)
                else: response_json['file_size'] = 0
            elif raw_data and response_json['file_size'] != len(raw_data):
                response_json['file_size'] = len(raw_data)

            # Encode JSON response to bytes
            json_data = json.dumps(response_json).encode('utf-8')

            # Send JSON header: 4-byte length followed by JSON data
            # '!I' means big-endian unsigned int (4 bytes)
            # This is the same format the client uses for requests
            header = struct.pack('!I', len(json_data))
            client_socket.sendall(header + json_data)

            if raw_data:

                # Send the actual data file as well; client will read exactly file_size bytes
                client_socket.sendall(raw_data)

                logger.info(f"Sent {len(raw_data)} bytes of binary data to client")

        except Exception as e:
            # If we can't even send an error response, log it and give up; This usually means the client disconnected unexpectedly
            logger.error(f"Error sending response to client: {e}")

    def receive_binary_data(self, client_socket, file_size):
        """
        Receive binary data from client

        Args:
            client_socket: Connected client socket
            file_size: Number of bytes to receive

        Returns:
            bytes: The received binary data
        """
        if file_size <= 0:
            return None

        binary_data = b''
        while len(binary_data) < file_size:
            bytes_needed = file_size - len(binary_data)
            chunk = client_socket.recv(min(bytes_needed, 8192))  # Receive in chunks
            if not chunk:
                raise ValueError("Connection closed prematurely while receiving binary data")
            binary_data += chunk

        logger.debug(f"Received {len(binary_data)} bytes of binary data")
        return binary_data

    def cleanup_session(self, session_id):
        """
        Remove a session from active sessions

        Args:
            session_id: The session ID to remove
        """
        with self.sessions_lock:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
                logger.info(f"Cleaned up session {session_id}")


    def handle_client(self, client_socket, address):
        """
        Handle a persistent client connection.

        **This runs in a thread and persists for the life cycle of the connection **

        This function processes multiple request-response cycles for a single client:
        1. Handle initial connection establishment and session ID assignment
        2. Process subsequent requests within the session
        3. Maintain connection until client disconnects or sends termination command
        4. Handle binary data reception from client

        Args:
            client_socket: Connected socket for this client
            address: Client's IP address and port tuple (e.g., ('127.0.0.1', 54321))
        """
        session_id = None
        persistent = False

        try:
            logger.info(f"New client connected from {address[0]}:{address[1]}")

            # Set socket timeout to prevent hanging on broken connections
            client_socket.settimeout(self.client_timeout)

            # Main loop for handling connection
            while not self.shutdown_requested:
                try:
                    # STEP 1: Receive JSON header (4 bytes containing message length)
                    header_data = client_socket.recv(4)
                    if len(header_data) != 4:
                        if len(header_data) == 0:
                            logger.info(f"Client {address[0]}:{address[1]} disconnected gracefully")
                            break
                        raise ValueError("Invalid header: expected 4 bytes, got " + str(len(header_data)))

                    # Unpack the header to get JSON message length
                    json_length = struct.unpack('!I', header_data)[0]
                    logger.debug(f"Expecting {json_length} bytes of JSON data")

                    # STEP 2: Receive the JSON message data
                    json_data = b''
                    while len(json_data) < json_length:
                        bytes_needed = json_length - len(json_data)
                        chunk = client_socket.recv(bytes_needed)
                        if not chunk:
                            raise ValueError("Connection closed prematurely while receiving JSON data")
                        json_data += chunk

                    logger.debug(f"Received {len(json_data)} bytes of JSON data")

                    # STEP 3: Parse and validate the JSON request
                    try:
                        request = json.loads(json_data.decode('utf-8'))
                        logger.debug(f"Parsed JSON request from {address[0]}:{address[1]}")

                        # Add client connection info to request
                        request['client_address'] = address[0]
                        request['client_port'] = address[1]

                        # STEP 4: Handle binary data if specified
                        client_binary_data = None
                        file_size = request.get('file_size', 0)
                        if file_size > 0:
                            client_binary_data = self.receive_binary_data(client_socket, file_size)

                        # store if this is a persistent connection or a simple request / response
                        persistent = request.get('persistent', False) # False - if we assume true and this is not persistent we could hold this open longer than we should, so if persistent is missing - this is a one and done

                        # STEP 5: Handle session management
                        client_session_id = request.get('sessionID', '')
                        request_id = request.get('requestID', '') # This can change for every call - its more of a passthrough

                        if not persistent:
                            #if this isnt persistent, set sessionID - this may be a passthrough. even though its a request/response situation, the client may need the sessionID on its end, so include it
                            # it doesnt even matter if the sessionID is in use by this service - it will never be used or checked for anything else
                            if client_session_id != '':
                                session_id = client_session_id
                            else:
                                session_id = self.generate_session_id()

                            request['sessionID'] = session_id
                            logger.info(f"Transitory connection from {address[0]}:{address[1]} assigned sessionID {session_id} ")
                        elif persistent and request.get('command') == 'establish_connection':
                            # Store session info - blocks until the lock is removed, claims the lock for itself, updates, then releases the lock
                            with self.sessions_lock:
                                if session_id:
                                    logger.info(f"Persistent connection from {address[0]}:{address[1]} tried to establish twice. SessionID {session_id}.")

                                    # if the sessionID has been set, it means we established the connection earlier - send back an error
                                    error_response = {
                                        'success': False,
                                        'type': 'connection',
                                        'message': f"establish_connection run twice.",
                                        'sessionID': client_session_id,
                                        'requestID': request_id,
                                        'client_address': address[0],
                                        'client_port': address[1],
                                        'file_size': 0
                                    }
                                    self.send_response(client_socket, error_response, None)
                                    continue
                                elif client_session_id != '' and client_session_id not in self.active_sessions:
                                    # if the client_session_id is not empty and its not in self.active_sessions, we can use it
                                    session_id = client_session_id
                                    logger.info(f"New persistent connection from {address[0]}:{address[1]} requested sessionID {client_session_id} - granted.")
                                elif client_session_id == '':
                                    # Initial connection - and session_id is guaranteed to be blank here. Generate new session ID
                                    session_id = self.generate_session_id()
                                    logger.info(f"New persistent connection from {address[0]}:{address[1]} given sessionID {session_id}.")
                                else:
                                    # client_session_id != '' but it was already in the list of sessionIDs - a big no-no
                                    logger.warning(f"New persistent session requested {session_id} for {address[0]}:{address[1]}, but that sessionID already in use.")
                                    error_response = {
                                        'success': False,
                                        'type': 'connection',
                                        'message': f"SessionID in use.",
                                        'sessionID': client_session_id,
                                        'requestID': request_id,
                                        'client_address': address[0],
                                        'client_port': address[1],
                                        'file_size': 0
                                    }
                                    self.send_response(client_socket, error_response, None)
                                    continue

                                # save the active session info
                                self.active_sessions[session_id] = {
                                    'address': address,
                                    'socket': client_socket,
                                    'created': time.time()
                                }

                            request['sessionID'] = session_id

                            logger.info(f"Established new persistent session {session_id} for {address[0]}:{address[1]}")
                            success_response = {
                                'success': True,
                                'type': 'connection',
                                'message': f"Connection established",
                                'sessionID': session_id,
                                'requestID': request_id,
                                'client_address': address[0],
                                'client_port': address[1],
                                'file_size': 0
                            }
                            self.send_response(client_socket, success_response, None)
                            continue

                        elif persistent and client_session_id:
                            # Existing session - validate it
                            with self.sessions_lock:
                                if client_session_id not in self.active_sessions:
                                    # Invalid session
                                    error_response = {
                                        'success': False,
                                        'type': 'error',
                                        'message': f'Invalid session ID: {client_session_id}. The session may no longer be tracked.',
                                        'sessionID': '',
                                        'requestID': request_id,
                                        'client_address': address[0],
                                        'client_port': address[1],
                                        'file_size': 0
                                    }
                                    self.send_response(client_socket, error_response, None)
                                    continue
                                else:
                                    # client_session_id is in self.active_sessions
                                    trusted_socket = self.active_sessions[client_session_id]['socket']
                                    if trusted_socket != client_socket:
                                        #someone may be trying to hijack this session
                                        error_response = {
                                            'success': False,
                                            'type': 'error',
                                            'message': f'You are being a naughty boy. Logged.',
                                            'sessionID': '',
                                            'requestID': request_id,
                                            'client_address': address[0],
                                            'client_port': address[1],
                                            'file_size': 0
                                        }
                                        self.send_response(client_socket, error_response, None)

                                        logger.warning(f"Someone tried claiming session ID {client_session_id}, but the sockets do not match. client_address {address[0]}")
                                        continue



                        elif persistent:
                            # No session ID and not establishing connection
                            error_response = {
                                'success': False,
                                'type': 'error',
                                'message': f'Session ID required for all commands except establish_connection.',
                                'sessionID': '',
                                'requestID': request_id,
                                'client_address': address[0],
                                'client_port': address[1],
                                'file_size': 0
                            }
                            self.send_response(client_socket, error_response, None)
                            continue

                        # at this point, both persistent and transient connections are good to more forward

                        # STEP 6: Check for termination command
                        if request.get('command') == 'terminate_session':
                            termination_response = {
                                'success': True,
                                'type': 'connection',
                                'message': 'Session terminated successfully',
                                'sessionID': session_id,
                                'requestID': request_id,
                                'client_address': address[0],
                                'client_port': address[1],
                                'file_size': 0
                            }

                            # If the session_id exists and additional_shutdown is set, run the shutdown
                            if session_id and self.additional_shutdown is not None:
                                self.additional_shutdown(session_id)

                            self.send_response(client_socket, termination_response, None)
                            logger.info(f"Session {session_id} terminated by client")
                            break

                        # STEP 7: Process request through callback function
                        if self.additional_client_functionality is not None:
                            request['_client_info'] = {'socket': client_socket, 'address': address} # necessary, as the callback is not always simple and straightforward if we need to make asynchronous / complex calls
                            json_dict, raw_data = self.additional_client_functionality(request, client_binary_data)

                            if self.synchronous:
                                # Ensure required fields are present
                                if 'sessionID' not in json_dict:
                                    json_dict['sessionID'] = session_id
                                if 'requestID' not in json_dict:
                                    json_dict['requestID'] = request_id
                                if 'client_address' not in json_dict:
                                    json_dict['client_address'] = address[0]
                                if 'client_port' not in json_dict:
                                    json_dict['client_port'] = address[1]

                                # Send response back to client
                                self.send_response(client_socket, json_dict, raw_data)

                            if not persistent:
                                # if this is not persistent, we did everything we had to do - exit
                                logger.info(f"Transitory session {session_id} complete")
                                break

                        else:
                            # No callback function - send basic success response
                            default_response = {
                                'success': True,
                                'type': 'connection',
                                'message': 'Request processed (no additional functionality configured)',
                                'sessionID': session_id,
                                'client_address': address[0],
                                'client_port': address[1],
                                'file_size': 0
                            }
                            self.send_response(client_socket, default_response, None)
                            if not persistent:
                                # if this is not persistent, we did everything we had to do - exit
                                logger.info(f"Transitory session {session_id} complete")
                                break

                        logger.info(f"Successfully processed request from {address[0]} (Session: {session_id})")

                    except (json.JSONDecodeError, KeyError) as e:
                        return_json = {
                            'success': False,
                            'type': 'error',
                            'message': f"Invalid JSON request: {e}",
                            'sessionID': session_id or '',
                            'client_address': address[0],
                            'client_port': address[1],
                            'file_size': 0
                        }
                        self.send_response(client_socket, return_json, None)

                    except Exception as e:
                        # Request processing failed - send error response to client
                        error_msg = f"Server Error: {str(e)}"
                        logger.error(f"Request from {address[0]} failed: {error_msg}")
                        return_json = {
                            'success': False,
                            'type': 'error',
                            'message': error_msg,
                            'sessionID': session_id or '',
                            'client_address': address[0],
                            'client_port': address[1],
                            'file_size': 0
                        }
                        self.send_response(client_socket, return_json, None)

                except socket.timeout:
                    # Client has been inactive for too long
                    logger.info(f"Client {address[0]}:{address[1]} timed out (Session: {session_id})")
                    break

                except Exception as e:
                    if "Connection reset by peer" in str(e) or "Broken pipe" in str(e):
                        logger.info(f"Client {address[0]}:{address[1]} disconnected unexpectedly (Session: {session_id})")
                        break
                    else:
                        logger.error(f"Error in client handling loop: {e}")
                        break

        except Exception as e:
            # Handle any other errors during client processing
            error_msg = f"Error processing client connection: {str(e)}"
            logger.error(f"Client {address[0]} error: {error_msg}")
            try:
                # Try to send error response if possible
                return_json = {
                    'success': False,
                    'type': 'error',
                    'message': error_msg,
                    'sessionID': session_id or '',
                    'client_address': address[0],
                    'client_port': address[1],
                    'file_size': 0
                }
                self.send_response(client_socket, return_json, None)
            except:
                # If even error response fails, just log and continue
                logger.error(f"Could not send error response to {address[0]}")

        finally:
            # Clean up connection and session
            if session_id:
                self.cleanup_session(session_id)

            try:
                client_socket.shutdown(socket.SHUT_RDWR)
            except:
                pass

            try:
                client_socket.close()
            except:
                pass

            logger.info(f"Connection with {address[0]}:{address[1]} closed (Session: {session_id})")

    def start_server(self):
        """
        Start the server and listen for connections

        This is the main server loop that:
        1. Creates and binds a server socket
        2. Sets up signal handlers for graceful shutdown
        3. Listens for incoming connections
        4. Handles each client connection in a separate thread for persistence
        5. Provides graceful shutdown on Ctrl-C without hanging

        This also has a timeout-based accept() loop that allows the server to respond to Ctrl-C quickly instead of hanging.
        """
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)   # Ctrl-C (SIGINT)
        signal.signal(signal.SIGTERM, self.signal_handler)  # Termination signal

        # Create TCP socket for the server
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Set socket option to reuse address
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Set socket timeout so accept() calls don't block forever
        server_socket.settimeout(1.0)  # 1 second timeout

        try:
            # Bind socket to the specified host and port
            server_socket.bind((self.host, self.port))

            # Start listening for connections
            server_socket.listen(5)

            logger.info(f"Server listening on {self.host}:{self.port}")
            logger.info(f"Press Ctrl-C to stop the server")

            # Main server loop - accept connections and handle them in threads
            while not self.shutdown_requested:
                try:
                    # Wait for a client to connect (with 1-second timeout)
                    client_socket, address = server_socket.accept()

                    # Handle this client's persistent connection in a separate thread
                    # This allows multiple clients to maintain persistent connections simultaneously
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address),
                        daemon=True  # Thread will be cleaned up when main program exits
                    )
                    client_thread.start()

                except socket.timeout:
                    # Accept timed out - this is normal, continue loop to check shutdown
                    continue

                except Exception as e:
                    # Log connection errors but keep server running unless shutdown requested
                    if not self.shutdown_requested:
                        logger.error(f"Error accepting connection: {e}")

        except Exception as e:
            # Handle server startup errors
            logger.error(f"Server error: {e}")
        finally:
            # Clean up server socket and active sessions
            logger.info("Shutting down server...")

            # Close all active sessions
            with self.sessions_lock:
                for session_id, session_info in list(self.active_sessions.items()):
                    try:
                        session_info['socket'].close()
                    except:
                        pass
                self.active_sessions.clear()

            try:
                server_socket.shutdown(socket.SHUT_RDWR)
            except:
                pass

            try:
                server_socket.close()
            except:
                pass

            logger.info("Server stopped successfully")