from typing import Dict, Any, Optional, Callable, Tuple
import signal
import logging
import json
import socket
import struct

"""
The idea behind this class is to handle all of the work of a simple server - bind to a host and port, accept client requests, respond, and close the connection (so its a simple get-request-handle-it-respond-shut down). The linchpin for this is it takes, as a variable, 
a 'callback function' (callback) method that can be called during a client request. As an example, I use this as an F5-TTS service, which accepts a voice identifier and text, and the server responds by changing that text to speech (using the given identified voice). The f5 object is defined, 
and then this class is quickly defined like so:  
```
        f5_model = f5.AmadeoF5(voice_samples_dir = voice_samples_dir)
        self.server = SimpleServer(host, port, additional_client_functionality = f5_model.handle_client_request)
```

As you can see, the f5_model's 'handle_client_request' is stored in 'additional_client_functionality' as a callback function; later, its called in this class' 'handle_client' like so:
```
                if self.additional_client_functionality is not None:
                    json_dict, raw_data = self.additional_client_functionality(request)
```

In this example, this method actually performs the text-to-speech work, but...it could do anything, really.
 
The client and server use JSON for communication, and allow for byte data to be tacked on just after the JSON (.wav, .mp3. movies, pictures, even text files, if you encode in UTF-8 a la `"Hello world!".encode('utf-8')`). There is no structure to the JSON the server accepts from the client;
the server will simply convert the JSON to a dictionary, tack on 'client_address' and 'client_port' (for logging purposes in the callback), and then send the dictionary to the callback function. The callback function MUST return a tuple (dictionary, binary), where the 'binary' is the raw data 
of a media file etc (it can also easily be 'None' if there is no data to be sent). This dictionary DOES have a few required fields:
{
    'success': True/False,   # boolean representing if the call was successful
    'message': '',  # any additional message (or error); does not have to be populated, but it must exist
    'file_size': 0  # Integer representing the number of bytes to expect from the raw data (0 if not used). IF you send data and do not set this to len(raw_data), you probably do not need to 
                    worry - its set in this code. That said its probably best practice to set this.
} 

This dictionary is converted to JSON, converted to binary, and then the byte data from any media file is tacked onto the end (if it exists), and then the data is sent to the client.

The server will run indefinitely, servicing clients, until Ctrl-C is pressed, a SIGTERM is sent, etc to stop it (or it is otherwise killed).   
"""

# Configure logging to show timestamps and log levels
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

class SimpleServer:

    def __init__(self, host='localhost', port=8888, additional_client_functionality: Optional[Callable[[Dict[str, Any]], Tuple[Dict[str, Any], bytes]]]  = None):
        """
        Initialize the server

        Args:
            host: IP address to bind to (default: localhost)
            port: Port number to listen on (default: 8888)
            additional_client_functionality: A method that will be called from this class to handle any additional functionality surrounding a client request that will need to be handled by this server.
                                            The method expects a dictionary as its only arg, and should return a dictionary (that will be turned into JSON) and byte data (if applicable)
                                            The dictionary being sent to the method is arbitrary (whatever it needs); the dictionary returned by the method _MUST_ have at least these fields:
                                            {
                                                'success': True/False,   # boolean representing if the call was successful
                                                'message': '',  # any additional message (or error); does not have to be populated, but it must exist
                                                'file_size': 0  # Integer representing the number of bytes to expect from the raw data (0 if not used). IF you send data and do not set this to len(raw_data), you probably do not need to
                                                                worry - its set in this code. That said its probably best practice to set this.
                                            }
        """
        self.host = host
        self.port = port

        # Flag to control server shutdown
        # This is set by the signal handler when Ctrl-C is pressed
        self.shutdown_requested = False

        if additional_client_functionality is not None:
            self.additional_client_functionality = additional_client_functionality
        else:
            logger.warning(f"No additional_client_functionality set; this will be a purposeless server.")

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


    def reset_additional_client_functionality(self, additional_client_functionality: Optional[Callable[[Dict[str, Any]], Tuple[Dict[str, Any], bytes]]]):
        """
        Re-sets additional_client_functionality

        Args:
            additional_client_functionality: A method that will be called from this class to handle any additional functionality surrounding a client request that will need to be handled by this server.
                                            The method expects a dictionary as its only arg, and should return a dictionary (that will be turned into JSON) and byte data (if applicable)
                                            The dictionary being sent to the method is arbitrary (whatever it needs); the dictionary returned by the method _MUST_ have at least these fields:
                                            {
                                                'success': True/False,   # boolean representing if the call was successful
                                                'message': '',  # any additional message (or error); does not have to be populated, but it must exist
                                                'file_size': 0  # Integer representing the number of bytes to expect from the raw data (0 if not used). IF you send data and do not set this to len(raw_data), you probably do not need to
                                                                worry - its set in this code. That said its probably best practice to set this.
                                            }
        """
        self.additional_client_functionality = additional_client_functionality



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

            if not response_json['file_size']:
                if raw_data: response_json['file_size'] = len(raw_data)
                else: response_json['file_size'] = 0
            elif response_json['file_size'] != len(raw_data): response_json['file_size'] = len(raw_data)

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

                logger.info(f"Sent {len(raw_data)} bytes of audio data to client")

        except Exception as e:
            # If we can't even send an error response, log it and give up; This usually means the client disconnected unexpectedly
            logger.error(f"Error sending response to client: {e}")


    def handle_client(self, client_socket, address):
        """
        Handle a single client request

        This function processes one complete request-response cycle:
        1. Receive JSON request from client
        2. Parse and validate the request
        3. Call the callback function, getting any response and / or byte file ready
        4. Send response back to client
        5. Clean up connection

        Each client is handled synchronously (one at a time). For concurrent
        clients, you would need to use threading or async programming.

        Args:
            client_socket: Connected socket for this client
            address: Client's IP address and port tuple (e.g., ('127.0.0.1', 54321))
        """
        try:
            logger.info(f"New client connected from {address[0]}:{address[1]}")

            # Set socket timeout to prevent hanging on broken connections
            # If no data received for 60 seconds, the operation will timeout
            # This prevents clients from hanging the server indefinitely
            client_socket.settimeout(60)

            # STEP 1: Receive JSON header (4 bytes containing message length)
            # The client sends this first so we know how much data to expect
            header_data = client_socket.recv(4)
            if len(header_data) != 4:
                raise ValueError("Invalid header: expected 4 bytes, got " + str(len(header_data)))

            # Unpack the header to get JSON message length
            # '!I' means big-endian unsigned integer (4 bytes)
            # This gives us the number of bytes in the JSON message
            json_length = struct.unpack('!I', header_data)[0]
            logger.debug(f"Expecting {json_length} bytes of JSON data")

            # STEP 2: Receive the JSON message data
            json_data = b''  # Start with empty bytes buffer
            while len(json_data) < json_length:
                # Receive remaining bytes
                # We might not get all the data in one recv() call
                bytes_needed = json_length - len(json_data)
                chunk = client_socket.recv(bytes_needed)
                if not chunk:
                    # Connection closed before all data received
                    raise ValueError("Connection closed prematurely while receiving JSON data")
                json_data += chunk

            logger.debug(f"Received {len(json_data)} bytes of JSON data")

            # STEP 3: Parse and validate the JSON request
            try:
                # Decode bytes to UTF-8 string and parse as JSON
                request = json.loads(json_data.decode('utf-8'))
                logger.debug(f"Parsed JSON request from {address[0]}:{address[1]}")

                request['client_address'] = address[0]
                request['client_port'] = address[1]

                if self.additional_client_functionality is not None:
                    json_dict, raw_data = self.additional_client_functionality(request)

                    # Send success response with audio data to client
                    self.send_response(client_socket, json_dict, raw_data)

                logger.info(f"Successfully processed request from {address[0]}")
            except (json.JSONDecodeError, KeyError) as e:

                return_json = { 'success': False, 'message': f"Invalid JSON request: {e}" }
                self.send_response(client_socket, return_json, None)

                raise ValueError()

            except Exception as e:
                # Speech generation failed - send error response to client
                error_msg = f"Server Error: {str(e)}"
                logger.error(f"Request from {address[0]} failed: {error_msg}")
                return_json = { 'success': False, 'message': error_msg }
                self.send_response(client_socket, return_json, None)

        except Exception as e:
            # Handle any other errors during client processing
            error_msg = f"Error processing client request: {str(e)}"
            logger.error(f"Client {address[0]} error: {error_msg}")
            return_json = { 'success': False, 'message': error_msg }
            try:
                # Try to send error response if possible
                self.send_response(client_socket, return_json, None)
            except:
                # If even error response fails, just log and continue
                # This usually means the client disconnected
                logger.error(f"Could not send error response to {address[0]}")

        finally:
            # Clean up connection (always executed, even if exceptions occurred)

            try:
                # Gracefully shutdown the socket connection
                # SHUT_RDWR means shutdown both read and write directions
                client_socket.shutdown(socket.SHUT_RDWR)
            except:
                # Shutdown might fail if connection already broken
                pass

            try:
                # Close the socket to free up system resources
                client_socket.close()
            except:
                # Close might fail if already closed
                pass

            logger.info(f"Connection with {address[0]}:{address[1]} closed")


    def start_server(self):
        """
        Start the server and listen for connections

        This is the main server loop that:
        1. Creates and binds a server socket
        2. Sets up signal handlers for graceful shutdown
        3. Listens for incoming connections
        4. Handles each client connection sequentially
        5. Provides graceful shutdown on Ctrl-C without hanging

        This also has a timeout-based accept() loop that allows the server to respond to Ctrl-C quickly instead of hanging.
        """
        # Set up signal handlers for graceful shutdown
        # These handle Ctrl-C and kill commands without hanging
        signal.signal(signal.SIGINT, self.signal_handler)   # Ctrl-C (SIGINT)
        signal.signal(signal.SIGTERM, self.signal_handler)  # Termination signal

        # Create TCP socket for the server
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Set socket option to reuse address
        # This prevents "Address already in use" errors when restarting the server quickly
        # Without this, you might have to wait 30-60 seconds after stopping the server
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Set socket timeout so accept() calls don't block forever
        # This is the KEY to fixing the Ctrl-C hanging issue!
        # accept() will timeout every 1 second, giving us a chance to check for shutdown
        server_socket.settimeout(1.0)  # 1 second timeout

        try:
            # Bind socket to the specified host and port
            server_socket.bind((self.host, self.port))

            # Start listening for connections
            # Backlog of 5 means up to 5 pending connections can wait in the queue
            server_socket.listen(5)

            logger.info(f"Server listening on {self.host}:{self.port}")
            logger.info(f"Press Ctrl-C to stop the server")

            # Main server loop - handle connections one at a time
            # This loop continues until shutdown_requested becomes True
            while not self.shutdown_requested:
                try:
                    # Wait for a client to connect (with 1-second timeout)
                    # The timeout allows us to check shutdown_requested periodically
                    # Without timeout, this would block forever and ignore Ctrl-C
                    client_socket, address = server_socket.accept()

                    # Handle this client's request synchronously
                    # Note: This processes one client at a time
                    # For concurrent clients, you'd use threading:
                    # threading.Thread(target=self.handle_client, args=(client_socket, address)).start()
                    self.handle_client(client_socket, address)

                except socket.timeout:
                    # Accept timed out after 1 second - this is normal and expected
                    # This gives us a chance to check if shutdown was requested
                    # No action needed, just continue the loop
                    continue

                except Exception as e:
                    # Log connection errors but keep server running
                    # unless shutdown was requested
                    if not self.shutdown_requested:
                        logger.error(f"Error accepting connection: {e}")

        except Exception as e:
            # Handle server startup errors
            logger.error(f"Server error: {e}")
        finally:
            # Clean up server socket (always executed)
            logger.info("Shutting down server...")

            try:
                # Gracefully shutdown server socket
                server_socket.shutdown(socket.SHUT_RDWR)
            except:
                # Might fail if not connected
                pass

            try:
                # Close server socket to free up the port
                server_socket.close()
            except:
                # Might fail if already closed
                pass

            logger.info("Server stopped successfully")