import logging
import sys
from amadeo_utils.server.amadeo_server import AmadeoServer
from amadeo_utils.logging_utils import add_log_file
from amadeo_utils.ai.llm.llama.ToolStream import ToolStream

# Configure logging to show timestamps and log levels
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)


class AmadeoAgentServer:
    """
    Serves ToolStream - the tool-calling LLM family (CS-21) - over the same socket protocol as the role-play and
    knowledge-base servers, so LlamaStreamClient can talk to it with '--mode agent'.

    Named an AGENT server (it was tool_server.py until 2026-09-25): the model decides for itself which tools to use,
    when, and in what order, over several rounds - it is not a single tool call.
    """
    def __init__(self, argsDict: dict):
        self.args_dict = argsDict

        self.model = ToolStream(argsDict)
        # client_idle_timeout_seconds 0 = never close an idle client (the always-on assistant); AmadeoServer then uses
        # TCP keepalive to notice a client that vanished instead.
        self.server = AmadeoServer(argsDict['host'], argsDict['port'],
                                   client_timeout=argsDict.get('client_idle_timeout_seconds', AmadeoServer.CLIENT_TIMEOUT),
                                   additional_client_functionality = self.model.handle_client_request,
                                   additional_shutdown = self.model.remove_session)


if __name__ == "__main__":

    argsDict = ToolStream.get_args_dict()
    if not argsDict:
        sys.exit(1)
    # The log goes to the screen, and also to 'log_file' if the config names one (see amadeo_utils.logging_utils)
    add_log_file(argsDict.get('log_file'))
    server = AmadeoAgentServer(argsDict)
    server.server.start_server()
