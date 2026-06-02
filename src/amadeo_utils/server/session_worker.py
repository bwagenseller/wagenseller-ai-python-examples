import logging
import queue
import threading
import socket
import json
import struct
import copy
from typing import Dict, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor

# Configure logging to show timestamps and log levels
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

"""
This facilitates running a thread per sessionID. It allows for an arbitrary number of callback functions to be called from this object - just keep in mind, the callback methods set MUST be made sure to be thread-safe or static. 

Commands are executed asynchronously in a thread pool, allowing multiple commands to run concurrently per session.

Args:
    session_id - The sessionID. This should not change for the entire session.
    client_socket - The actual socket that we need to communicate with the client
    address - A Tuple representing a hostname and port for the client.  
    pipeline - if this worker is for a specific pipeline, list this here. Can only be set in the constructor, as I dont want to put a lock on this unless absolutely necessary.  
    parent_server - This allows for the worker to interact with the parent object, being able to call its methods. GREAT caution with this as you need to make sure everything is thread-safe. To set this in the parent, just say parent_server=self 
    command_handlers - a dictionary of a set of commands (a key, str format) and callable methods. This is what the worker will 'do'
    max_workers - Maximum number of concurrent commands that can run for this session (default: 3)
      

To add a job, you need to create a dictionary and have AT LEAST a 'command' entry that will link up to an entry in command_handlers; the entire dictionary will be sent to the callback function found in command_handlers. An example
of a job is below - it attaches the required command, the request from a client, the sessionID, and some raw audio data for processing (for ASR):   
job = {
    'command': 'conversational_pipeline', 
    'sessionID': session_id,
    'audio_data': client_binary_data,
    'request': request
}


      
Usage: 

        
# Register shared commands once

def handle_ping(worker, job):
    print(f"Pong from session {worker.session_id}")

def handle_status(worker, job):
    print(f"Session {worker.session_id} is running: {worker.running}")

# Register these globally - all workers get them. You can do this before one instance is made
SessionWorker.register_global_handler('ping', handle_ping)
SessionWorker.register_global_handler('status', handle_status)

# Now create workers with session-specific handlers
session_handlers = {
    'conversational_pipeline': handle_conversational_pipeline,
    'file_upload': handle_file_upload,
}

# create the worker and also start it
worker = SessionWorker(
    session_id="123",
    client_socket=socket,
    address=addr,
    parent_server=self,
    command_handlers=session_handlers,  # Instance-specific only
    max_workers=5  # Allow up to 5 concurrent commands
)

# This worker automatically has: shutdown, ping, status, conversational_pipeline, file_upload


# add a job

job = {
    'command': 'conversational_pipeline', 
    'sessionID': session_id,
    'audio_data': client_binary_data,
    'request': request
}

worker.add_work(job) # job will be processed asynchronously

# shutdown

worker.shutdown() # also shuts down client socket and waits for active commands to complete

"""
class SessionWorker:
    # Class-level registry for shared commands
    _global_handlers = {}
    _handlers_initialized = False

    @classmethod
    def register_global_handler(cls, command, callback):
        """Register a handler that ALL workers will have"""
        cls._global_handlers[command] = callback

    @classmethod
    def _ensure_handlers_initialized(cls):
        """Initialize handlers if not already done"""
        if not cls._handlers_initialized:
            cls._global_handlers['shutdown'] = cls._handle_shutdown
            cls._handlers_initialized = True

    @classmethod
    def _handle_shutdown(cls, worker, job):
        """Default shutdown handler"""
        logger.info(f"Shutdown command received for session {worker.session_id}")
        worker.running = False

    def __init__(self, session_id: str, client_socket: socket.socket, address: tuple[str, int], parent_server: 'ParentServer', pipeline = '', command_handlers: Optional[Dict[str, Callable[['SessionWorker', Dict[str, Any]], None]]] = None, max_workers: int = 3):
        """
        Args:
            session_id - The sessionID. This should not change for the entire session.
            client_socket - The actual socket that we need to communicate with the client
            address - A Tuple representing a hostname and port for the client.
            pipeline - if this worker is for a specific pipeline, list this here. Can only be set in the constructor, as I dont want to put a lock on this unless absolutely necessary.
            parent_server - This allows for the worker to interact with the parent object, being able to call its methods. GREAT caution with this as you need to make sure everything is thread-safe
            command_handlers - a dictionary of a set of commands (a key, str format) and callable methods. This is what the worker will 'do'
            max_workers - Maximum number of concurrent commands that can run for this session

        """
        self._ensure_handlers_initialized()

        self.session_id = session_id
        self.client_socket = client_socket
        self.address = address
        self.pipeline = pipeline
        self.parent_server = parent_server
        self.work_queue = queue.Queue()
        self.running = True

        # Instance-level results - each worker gets its own
        # This is meant to store odds and ends
        self.backpack = {}
        self.backpack_lock = threading.RLock()

        # Thread pool for async command execution
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"session_{session_id}_cmd"
        )
        self.active_futures = set()
        self.futures_lock = threading.Lock()

        # Start with global handlers, then add instance-specific ones
        self.command_handlers = self._global_handlers.copy()
        if command_handlers:
            self.command_handlers.update(command_handlers)

        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

        logger.info(f"Worker for ID {self.session_id} started.")

    def _worker_loop(self):
        try:
            while self.running:
                try:
                    job = self.work_queue.get(timeout=1)
                    command = job.get('command')

                    if command == 'shutdown':
                        logger.info(f"Shutdown command received for session {self.session_id}")
                        break  # Exit immediately on shutdown
                    elif command in self.command_handlers:
                        # Submit command for async execution
                        future = self.executor.submit(
                            self._execute_command_safely,
                            command, job
                        )

                        # Track the future
                        with self.futures_lock:
                            self.active_futures.add(future)

                        # Clean up completed futures periodically
                        self._cleanup_completed_futures()

                    else:
                        logger.warning(f"Unknown command: {command}")

                except queue.Empty:
                    # Periodically clean up completed futures even when no new jobs
                    self._cleanup_completed_futures()
                    continue
                except Exception as e:
                    logger.error(f"Session {self.session_id} worker error: {e}")
        finally:
            # Always clean up when exiting the loop
            self._cleanup()

    def _execute_command_safely(self, command: str, job: Dict[str, Any]):
        """Execute command with error handling in thread pool"""
        try:
            logger.debug(f"Starting async execution of '{command}' for session {self.session_id}")
            self.command_handlers[command](self, job)
            logger.debug(f"Completed async execution of '{command}' for session {self.session_id}")
        except Exception as e:
            logger.error(f"Error executing command '{command}' for session {self.session_id}: {e}")
            # BRENT
        finally:
            # Remove this future from active set when done
            current_future = None
            with self.futures_lock:
                # Find the future that corresponds to this execution
                for future in self.active_futures.copy():
                    if future.done():
                        try:
                            # Check if this is our future by seeing if it's done and matches our thread
                            self.active_futures.discard(future)
                        except:
                            pass

    def _cleanup_completed_futures(self):
        """Remove completed futures from tracking set"""
        with self.futures_lock:
            completed = {f for f in self.active_futures if f.done()}
            self.active_futures -= completed

            if completed:
                logger.debug(f"Cleaned up {len(completed)} completed futures for session {self.session_id}")

    def get_active_command_count(self) -> int:
        """Get the number of currently running commands"""
        with self.futures_lock:
            return len([f for f in self.active_futures if not f.done()])

    def get_pipeline(self):
        """No lock - it can only be set in the constructor"""
        return self.pipeline

    def add_work(self, job):
        """Add work to this session's queue"""
        self.work_queue.put(job)

    def save_in_backpack(self, key: str, item: Dict) -> None:
        """Thread-safe save a backpack item for this session"""
        with self.backpack_lock:
            self.backpack[key] = item

    def get_from_backpack(self, key: str) -> Any:
        """Thread-safe get from backpack - returns a copy"""
        with self.backpack_lock:
            item = self.backpack.get(key)
            if isinstance(item, bytes):
                return item[:]  # Slice creates a copy of bytes
            elif hasattr(item, 'copy'):
                return item.copy()  # For dicts, lists, etc.
            else:
                if item is not None:
                    return copy.copy(item)  # Works for most types
                return None


    def get_all_backpack_items(self) -> Dict:
        """Thread-safe get all backpack items - returns a copy"""
        with self.backpack_lock:
            return {k: v.copy() for k, v in self.backpack.items()}

    def shutdown(self, wait_for_commands: bool = True, timeout: int = 30):
        """Gracefully shutdown this worker"""
        logger.info(f"Initiating shutdown for worker for session {self.session_id}")

        # Check if we're being called from within the thread pool
        current_thread_name = threading.current_thread().name
        is_thread_pool_thread = f"session_{self.session_id}_cmd" in current_thread_name

        # Get count of active commands
        active_count = self.get_active_command_count()
        if active_count > 0 and wait_for_commands:
            logger.info(f"Waiting for {active_count} active commands to complete for session {self.session_id}")

        self.add_work({'command': 'shutdown'})  # Trigger shutdown through normal job processing

        # Wait for main worker thread to finish - but only if we're not IN that thread OR a thread pool thread
        if self.thread and self.thread.is_alive():
            if threading.current_thread() != self.thread and not is_thread_pool_thread:
                self.thread.join(timeout=5)
                if self.thread.is_alive():
                    logger.warning(f"Session {self.session_id} main thread didn't shut down cleanly")
            else:
                logger.debug(f"Shutdown called from worker thread or thread pool, skipping join")

        # Shutdown thread pool - but only if we're not running IN the thread pool
        if hasattr(self, 'executor') and not is_thread_pool_thread:
            logger.info(f"Shutting down thread pool for session {self.session_id}")
            if wait_for_commands:
                self.executor.shutdown(wait=True)
            else:
                self.executor.shutdown(wait=False)
                # Cancel remaining futures if not waiting
                with self.futures_lock:
                    for future in self.active_futures:
                        future.cancel()
        elif is_thread_pool_thread:
            logger.debug(f"Shutdown called from thread pool thread, deferring executor shutdown")
            # Just set running to False and let the main thread handle cleanup
            self.running = False

    def _cleanup(self):
        """Clean up resources when worker shuts down"""
        # We used to clean up sockets here, but - a worker lasts only as long as the request to service the pipeline, and we do not want a worker closing the socket as that brings down the entire session
        pass
        """
        try:
            if hasattr(self, 'client_socket') and self.client_socket:
                logger.info(f"Closing socket for session {self.session_id}")
                self.client_socket.close()
        except Exception as e:
            logger.error(f"Error closing socket for session {self.session_id}: {e}")
        """

    def send_to_client(self, response_json, raw_data = None):
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
            self.client_socket.sendall(header + json_data)

            if raw_data:

                # Send the actual data file as well; client will read exactly file_size bytes
                self.client_socket.sendall(raw_data)

                logger.info(f"Sent {len(raw_data)} bytes of binary data to client")

        except Exception as e:
            # If we can't even send an error response, log it and give up; This usually means the client disconnected unexpectedly
            logger.error(f"Error sending response to client: {e}")