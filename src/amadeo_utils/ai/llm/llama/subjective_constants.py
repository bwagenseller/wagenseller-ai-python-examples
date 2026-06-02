
class SubjectiveConstants:

    """
    * EMBEDDING_MODEL
      * Provide the path to your DEDICATED EMBEDDING MODEL in GGUF format.
      * This model MUST produce fixed-size embeddings (e.g., Nomic Embed, BGE, E5 models).
      * Example: 'nomic-embed-text-v1.5.Q4_0.gguf'
    * MODEL
      * Provide the path to your GENERATIVE LLM in GGUF format.
      * This model will handle the chat responses and role-playing.
      * Example: 'llama-3-8b-instruct.Q4_K_M.gguf'

    NOTE: These are placeholder defaults. To use your own values without editing
    this file (and without committing private paths to git), create a sibling file
    named 'subjective_constants_local.py' defining a class 'SubjectiveConstantsLocal'
    with any of the attributes below. It is loaded automatically if present and is
    gitignored. See the override block at the bottom of this file.
    """
    BASE_MODEL_DIR = "/path/to/models/llama.cpp"
    BASE_EMBEDDING_DIR = "/path/to/models/llama.cpp/embedding_models"
    MODEL = "your-generative-model.gguf"
    EMBEDDING_MODEL = "nomic-embed-text-v1.5.Q5_K_M.gguf"

    BASE_PLAYER_NAME = ""
    SYSTEM_PROMPT_ID = "default"
    SYSTEM_PROMPT_FILE = "/path/to/system_prompt.txt"
    SYSTEM_PROMPT_DIR = "/path/to/prompts/"
    BASE_CONVO_DIR_OVERRIDE = "/path/to/chatHistory"
    BASE_CONVO_DIR = "/path/to/chatHistory"


# --- Optional local overrides ---
# Keep your real, private values in 'subjective_constants_local.py' (gitignored).
# Any public attribute it defines overrides the placeholder default above.
try:
    from amadeo_utils.ai.llm.llama.subjective_constants_local import SubjectiveConstantsLocal
    for _name in dir(SubjectiveConstantsLocal):
        if not _name.startswith("_"):
            setattr(SubjectiveConstants, _name, getattr(SubjectiveConstantsLocal, _name))
except ImportError:
    pass
