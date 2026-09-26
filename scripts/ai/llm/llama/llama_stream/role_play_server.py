import logging
from amadeo_utils.server.amadeo_server import AmadeoServer
from amadeo_utils.ai.llm.llama.RolePlayStream import RolePlayStream

# Configure logging to show timestamps and log levels
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

class RolePlayServer:
    """
    Constructor for RolePlayStream server
    """
    def __init__(self, argsDict: dict):
        self.args_dict = argsDict

        self.model = RolePlayStream(argsDict)
        # client_idle_timeout_seconds (server config): how long an idle client is kept; 0 = never, with TCP keepalive
        # noticing a client that vanished. A config without it keeps AmadeoServer's 300 seconds.
        self.server = AmadeoServer(argsDict['host'], argsDict['port'],
                                   client_timeout=argsDict.get('client_idle_timeout_seconds', AmadeoServer.CLIENT_TIMEOUT),
                                   additional_client_functionality = self.model.handle_client_request,
                                   additional_shutdown = self.model.remove_session)



if __name__ == "__main__":

    argsDict = RolePlayStream.get_args_dict()
    server = RolePlayServer(argsDict)
    server.server.start_server()