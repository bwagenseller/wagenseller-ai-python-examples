import logging
from amadeo_utils.ai.combined.conversational_ai.conversational_ai import ConversationalAiServer


"""
How the Server Works
Global Setup: The script starts by importing necessary libraries and setting up global variables for configuration, a queue.Queue for handling transcription jobs, and a threading.Lock to manage GPU access. The WhisperX model is loaded once at startup. ⚙️

Multithreading: The main loop (main()) continuously listens for incoming connections. When a client connects (s.accept()), it doesn't handle the client directly. Instead, it spawns a new thread (threading.Thread) to execute the handle_client function. This allows the main thread to immediately go back to listening for more connections, fulfilling the requirement of handling multiple requests at once.

Session Management: Each client thread generates a unique sessionID using uuid and sends it to the client. This ID is then used to track the session and its resources. A global sessions dictionary could be used to manage this, but the handle_client function here keeps the connection object (conn) in its local scope for simplicity and to ensure resources are properly released when the thread exits.

JSON Communication: The send_json and receive_json helper functions are crucial for handling communication. They implement a simple protocol where each JSON message is prefixed with a 4-byte header indicating the message's total length. This prevents the common problem of reading incomplete messages from the socket.

Handling a Busy GPU: The transcription_queue is the core of this feature. When a client sends an transcribe_audio command, the handle_client thread does not start the transcription immediately. Instead, it places the job (session ID, audio data, and connection object) into the queue. A separate, dedicated worker thread (worker_thread()) constantly monitors this queue. When it finds a job, it acquires the gpu_lock before starting the transcription. If the queue is full, the client is informed that the GPU is busy.

Resource Cleanup: When a client sends an end_session command, or if the connection is lost, the handle_client thread's while True loop breaks. The finally block then ensures that the socket connection is closed and any associated temporary files or session data are cleaned up, preventing resource leaks.



The server knows whether it's receiving JSON or an audio file by using a pre-defined communication protocol between the client and server. This protocol acts like a set of rules they both follow to interpret the data stream correctly. The core of this protocol is the JSON header.

The server doesn't magically "see" the data and know its type. Instead, it relies on the client to send a structured message that explicitly states what's coming next.

How the Protocol Works
The Server Always Expects a Header First: The server's receive_json function is designed to always read a 4-byte header first. This header is not part of the message content itself; it's a separate piece of data that tells the server how long the next message is.

The Header Dictates the Read Size: After reading the 4-byte header, the server knows exactly how many bytes to read next. It then loops to ensure it receives that exact number of bytes for the full message.

The Message Type is a JSON Field: The "magic" happens inside the JSON message itself. The client sends a JSON object with a specific field, such as "command". The value of this field tells the server what to do.

If the command is "transcribe_audio", the server knows that a specific type of data (in this case, the audio file) will be sent as the value for the "audio_data" field.

If the command is "end_session", the server knows the client is done and no more data is coming.

"""

# Configure logging to show timestamps and log levels
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

class ConversationalAiPipelineServer:
    """
    Constructor for ConversationalAiServer
    """
    def __init__(self, argsDict: dict):
        self.args_dict = argsDict

        self.server = ConversationalAiServer(argsDict)
        self.server.server.start_server()




if __name__ == "__main__":

    argsDict = ConversationalAiServer.get_args_dict_server()
    server = ConversationalAiPipelineServer(argsDict)