import logging
from amadeo_utils.server.amadeo_server import AmadeoServer
from amadeo_utils.ai.llm.llama.KnowledgeBaseStream import KnowledgeBaseStream

# Configure logging to show timestamps and log levels
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

class KnowledgeBaseServer:
    """
    Constructor for KnowledgeBaseStream server
    """
    def __init__(self, argsDict: dict):
        self.args_dict = argsDict

        self.model = KnowledgeBaseStream(argsDict)
        self.server = AmadeoServer(argsDict['host'], argsDict['port'], additional_client_functionality = self.model.handle_client_request, additional_shutdown = self.model.remove_session)



if __name__ == "__main__":

    argsDict = KnowledgeBaseStream.get_args_dict()
    server = KnowledgeBaseServer(argsDict)
    server.server.start_server()
