import numpy as np
import pandas as pd
import os
import json
import io
from typing import List, Tuple
from threading import Lock
from pathlib import Path
from amadeo_utils.colored_text import ColoredText
from amadeo_utils.ai.llm.llama.llama_utils import LlamaUtils
from amadeo_utils.misc_utils.FileEncryption import AmadeoEncryption

# Ensure you have pyarrow and fastparquet installed for DataFrame persistence:
# pip install pyarrow fastparquet


class VectorDB:
    """
    A simple in-memory vector database designed to store text documents
    and their corresponding embeddings for LLMs. Initially developed for use with llama.cpp
    generated embeddings, the hope is it can be used for others (Claude.ai, etc).

    It uses pandas DataFrame for data storage and numpy for efficient
    vector operations and cosine similarity calculations.
    """

    VECTOR_DB_USER_REQUEST = "User Request: "
    VECTOR_DB_AGENT_RESPONSE = "\nAssistant Response: "
    ASSISTANT_RESPONSE = "Fascinating."
    USER_REQUEST = "Update me."

    def __init__(self, llm_embedder, embedding_gpu_lock: Lock, llm_generator, generating_gpu_lock: Lock, generating_model_type, save_dir, debug, passphrase="", salt_path = None):

        self.df = VectorDB.get_base_dataframe()

        self._expected_embedding_dim = None # New attribute to store the dimension of embeddings

        self.save_dir = save_dir
        self.passphrase = passphrase # if passphrase is the empty string, just do not use encryption

        # --- Persistence Helper Functions ---
        if self.passphrase:
            self.vector_db_file = os.path.join(self.save_dir, "vector_db.parquet.enc")
            self.chat_history_file = os.path.join(self.save_dir, "chat_history.json.enc")
        else:
            self.vector_db_file = os.path.join(self.save_dir, "vector_db.parquet")
            self.chat_history_file = os.path.join(self.save_dir, "chat_history.json")

        self.llm_embedder = llm_embedder
        self.llm_generator = llm_generator

        self.embedding_gpu_lock = embedding_gpu_lock
        self.generating_gpu_lock = generating_gpu_lock
        self.generating_model_type = generating_model_type

        self.debug = debug

        if self.passphrase:
            if salt_path is None:
                salt_path = os.path.join(self.save_dir, 'convo.salt')
            self.encryption = AmadeoEncryption(self.passphrase, salt_path=salt_path)

        if self.passphrase:
            # Validate passphrase before doing anything
            if not self.validate_passphrase():
                raise ValueError("Incorrect passphrase! Cannot decrypt existing files.")



    def _ensure_numpy_array(self, data: any, name: str) -> np.ndarray:
        """
        Internal helper to ensure data is a numpy array. Converts if necessary
        and flattens to 1D.

        Args:
            data (any): The input data to convert.
            name (str): A descriptive name for the data (e.g., "embedding", "query embedding").

        Returns:
            np.ndarray: The data converted to a 1D numpy array.
        """
        if not isinstance(data, np.ndarray):
            # print(f"Warning: {name} is not a numpy array. Converting.") # Suppress for cleaner output
            data = np.array(data)
        return data.flatten() # Ensure it's a 1D array for consistent operations

    #def add_document(self, text: str, embedding: np.ndarray, tokens: int):
    #find_token_count_llama3
    def add_document(self, user_text: str, assistant_text: str):
        """
        Adds a single document (text and its embedding) to the database.

        Args:
            text (str): The original text of the document.
            embedding (np.ndarray): The numpy array representing the embedding of the text.
                                    Expected to be a 1D array from llama.cpp.
        """


        user_tokens = self.find_token_count("user", user_text)
        assistant_tokens = self.find_token_count("assistant", assistant_text)
        combined_turn_text = f"{VectorDB.VECTOR_DB_USER_REQUEST}{user_text}{VectorDB.VECTOR_DB_AGENT_RESPONSE}{assistant_text}"

        with self.embedding_gpu_lock:
            user_embedding = self._ensure_numpy_array(np.array(self.llm_embedder.embed(f"{user_text}"), dtype=np.float32), "document embedding")
            assistant_embedding = self._ensure_numpy_array(np.array(self.llm_embedder.embed(f"{assistant_text}"), dtype=np.float32), "document embedding")
            combined_turn_embedding = self._ensure_numpy_array(np.array(self.llm_embedder.embed(f"{combined_turn_text}"), dtype=np.float32), "document embedding")


        # Check for consistent embedding dimensions upon first addition
        if self._expected_embedding_dim is None:
            self._expected_embedding_dim = combined_turn_embedding.shape[0]
            if self.debug: print(f"{ColoredText.CYAN_TEXT}VectorDB.add_document: Set expected embedding dimension to {self._expected_embedding_dim}.{ColoredText.END_TEXT}")
        elif (combined_turn_embedding.shape[0] != self._expected_embedding_dim) or (assistant_embedding.shape[0] != self._expected_embedding_dim) or (user_embedding.shape[0] != self._expected_embedding_dim):
            raise ValueError(
                f"Inconsistent embedding dimension. Expected {self._expected_embedding_dim}, "
                f"but received {user_embedding.shape[0]} for the user embedding, {assistant_embedding.shape[0]} for the assistant embedding, amd {combined_turn_embedding.shape[0]} for the combined embedding."
                "\nPlease ensure your embedding model consistently produces fixed-size embeddings."
            )


        new_row = pd.DataFrame([{'embedding': combined_turn_embedding, 'user_text': user_text, 'user_embedding': user_embedding, 'user_token_count': user_tokens, 'assistant_text': assistant_text, 'assistant_embedding': assistant_embedding, 'assistant_token_count': assistant_tokens}])
        # Use pandas.concat for appending new rows as recommended for newer pandas versions.
        # ignore_index=True ensures the new DataFrame's index is reset and appended correctly.
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        # print(f"Added document to DB: '{text[:50]}{'...' if len(text) > 50 else ''}'") # Keep silent for conversation turns

    def add_documents(self, user_texts: List[str], assistant_texts: List[str]):
        """
        Adds multiple documents (user texts and responses) to the database in a batch.

        Args:
            user_texts (List[str]): A list of original user_texts.
            assistant_texts (List[str]): A list of 'responses' by the assistant.

        Raises:
            ValueError: If the lengths of the user_texts and embeddings lists do not match,
                        or if embedding dimensions are inconsistent.
        """
        if len(user_texts) != len(assistant_texts):
            raise ValueError(f"Lengths of user_texts [{len(user_texts)}] and assistant_texts [{len(assistant_texts)}] lists must match.")

        new_data = []
        for i, (user_request, assistant_response) in enumerate(zip(user_texts, assistant_texts)):

            user_tokens = self.find_token_count("user", user_request)
            assistant_tokens = self.find_token_count("assistant", assistant_response)
            combined_turn_text = f"{VectorDB.VECTOR_DB_USER_REQUEST}{user_request}{VectorDB.VECTOR_DB_AGENT_RESPONSE}{assistant_response}"

            with self.embedding_gpu_lock:
                user_embedding = self._ensure_numpy_array(np.array(self.llm_embedder.embed(f"{user_request}"), dtype=np.float32), "document embedding")
                assistant_embedding = self._ensure_numpy_array(np.array(self.llm_embedder.embed(f"{assistant_response}"), dtype=np.float32), "document embedding")
                combined_turn_embedding = self._ensure_numpy_array(np.array(self.llm_embedder.embed(f"{combined_turn_text}"), dtype=np.float32), "document embedding")


            # Check for consistent embedding dimensions for batch add
            if self._expected_embedding_dim is None:
                self._expected_embedding_dim = user_embedding.shape[0]
                if self.debug: print(f"{ColoredText.CYAN_TEXT}VectorDB.add_document: Set expected embedding dimension to {self._expected_embedding_dim}.{ColoredText.END_TEXT}")

            elif user_embedding.shape[0] != self._expected_embedding_dim:
                raise ValueError(
                    f"Inconsistent embedding dimension in batch. Expected {self._expected_embedding_dim}, "
                    f"but received shape of {user_embedding.shape[0]} for user_embedding, shape of {assistant_embedding.shape[0]} for assistant_embedding, and shape of {combined_turn_embedding.shape[0]} for combined_turn_embedding."
                    "\nPlease ensure your embedding model consistently produces fixed-size embeddings."
                )

            new_data.append({'embedding': combined_turn_embedding, 'user_text': user_request, 'user_embedding': user_embedding, 'user_token_count': user_tokens, 'assistant_text': assistant_response, 'assistant_embedding': assistant_embedding, 'assistant_token_count': assistant_tokens})

        if new_data:
            new_df = pd.DataFrame(new_data)
            self.df = pd.concat([self.df, new_df], ignore_index=True)
            if self.debug: print(f"{ColoredText.CYAN_TEXT}VectorDB.add_document: Added {len(new_data)} documents to DB.{ColoredText.END_TEXT}")
        else:
            if self.debug: print(f"{ColoredText.CYAN_TEXT}VectorDB.add_document: Document add attempted, but no document could be added.{ColoredText.END_TEXT}")

    """
    Removes the last row from the vector database, but only if the DataFrame is not empty.
    """
    def strike_last_from_record(self):
        if not self.df.empty:  # Check if the DataFrame is NOT empty
            # Get the index label of the last row
            last_row_index = self.df.index[-1]

            user_text = self.df.loc[last_row_index, 'user_text']  # .loc is used for label-based indexing
            if self.debug: print(f"{ColoredText.CYAN_TEXT}Removed row {last_row_index} from vector database with User Request: {user_text}.{ColoredText.END_TEXT}")
            else: print(f"{ColoredText.CYAN_TEXT}Removed row {last_row_index} from vector database.{ColoredText.END_TEXT}")

            # Drop the last row by its index label, in-place
            self.df.drop(last_row_index, inplace=True)

        else:
            print(f"{ColoredText.CYAN_TEXT}DataFrame is empty, no row to remove.{ColoredText.END_TEXT}")



    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        r"""
        Calculates the cosine similarity between two numpy vectors.
        Cosine similarity is a measure of similarity between two non-zero vectors
        of an inner product space that measures the cosine of the angle between them.
        It is often used to compare documents in text analysis.

        Formula: $similarity = (A \cdot B) / (||A|| \cdot ||B||)$
        where:
        - $A \cdot B$ is the dot product of vectors A and B.
        - $||A||$ is the L2 norm (magnitude) of vector A.
        - $||B||$ is the L2 norm (magnitude) of vector B.

        Args:
            vec1 (np.ndarray): The first vector (query embedding or document embedding).
            vec2 (np.ndarray): The second vector (document embedding).

        Returns:
            float: The cosine similarity score, ranging from -1 (opposite) to 1 (identical).
                   Returns 0.0 if either vector has a zero magnitude to prevent division by zero.
        """
        # Ensure vectors are 1D arrays
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()

        # Added explicit dimension check here as a fallback/diagnostic.
        # The primary check is now in add_document/add_documents during ingestion.
        if vec1.shape[0] != vec2.shape[0]:
            raise ValueError(
                f"Embedding dimension mismatch during cosine similarity calculation. "
                f"Query embedding has shape {vec1.shape[0]}, "
                f"document embedding has shape {vec2.shape[0]}."
                "\nThis indicates an issue with the embeddings themselves, "
                "likely from the embedding model not producing consistent sizes. "
                "Ensure your embedding model outputs fixed-size embeddings."
            )

        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0  # Return 0 for zero vectors to avoid division by zero

        return dot_product / (norm_vec1 * norm_vec2)

    """
    A wrapper around the function that performs a similarity search to find the top 'k' most similar documents to the given query embedding.

    Args:
        query (str): The query; is probably almost always the user prompt, but it can be different, depending.
        ignore_user (bool): If we want to ignore the user request; this is useful if you want the recall of the vector database to ignore your prompt for the round found in the vector database. This saves tokens by inserting a small user 
            request (i.e. 'Update me.') and then having the full assistant response
        ignore_assistant (bool): If we want to ignore the assistant response; this is useful if you want the recall of the vector database to ignore the assistant's response for the round found in the vector database. This saves tokens by 
            inserting a small assistant response (i.e. 'Fascinating.') and then leaving the associated user request intact. Use this if the assistant is chatty and you want the assistant to fully focus on what YOU said.            
        k (int, optional): The number of top similar documents to retrieve. Defaults to 5.

    Returns:
        List[Tuple[str, str, int, str, int, float]]: A list of tuples, where each tuple contains (column_header, user_request, user_token_count, assistant_response, assistant_token_count, similarity_score) for the top 'k' documents. The list is sorted by similarity score in descending order.
                                 Returns an empty list if the database is empty.
    """
    def search(self, query: str, ignore_user: bool, ignore_assistant: bool, k: int = 5) -> List[Tuple[str, int, str, int, float]]:

        if self.df.empty:
            if self.debug: print(f"{ColoredText.CYAN_TEXT}VectorDB.search: Database is empty. No search can be performed.{ColoredText.END_TEXT}")
            return []

        with self.embedding_gpu_lock:
            query_embedding = self._ensure_numpy_array(np.array(self.llm_embedder.embed(f"{query}"), dtype=np.float32), "query embedding")

        # Crucial check: Ensure the query embedding dimension matches what's expected in the DB
        if self._expected_embedding_dim is not None and query_embedding.shape[0] != self._expected_embedding_dim:
            raise ValueError(
                f"Query embedding dimension mismatch. Expected {self._expected_embedding_dim}, "
                f"but received {query_embedding.shape[0]}."
                "\nPlease ensure your embedding model produces consistent embedding sizes "
                "for both initial documents and queries."
            )


        """
        This will compare the query / user request embedding to:
        * The embedding that represents the combined text of the user request / ai assistant response. Many vector databases use the combined text for the embedding.  
        * The embedding that represents the text of the user request. Its often the case where a better match is found with an isolated user request; this is especially true if the AI assistant 
            rambles on and on, polluting the context, OR if it asks unrelated questions. This option is best if you want the AI to focus more on what you say and less on what it says - so if you are using 
            it to collect your thoughts / make a decision based on a collection of items you wrote, this is for you.  
        * The embedding that represents the text of the AI assistant response. Its often the case where a better match is found with an isolated user request; this is especially true if the AI assistant is 
            building a world / story and you are just living in it. This option is best if you want the AI to focus more on the world it builds / the information it provides and less on what you say.
            
        Right now, it just uses all 3, taking the top k of each, and then orders those results from highest to lowest and sends them to you. you are welcome to alter how this works.
        """
        results = []
        results.extend(self._get_search_results('embedding', query_embedding, ignore_user, ignore_assistant, k))
        results.extend(self._get_search_results('user_embedding', query_embedding, ignore_user, ignore_assistant, k))
        results.extend(self._get_search_results('assistant_embedding', query_embedding, ignore_user, ignore_assistant, k))

        # re-sort the list by the 6th column (score)
        results.sort(key=lambda x: x[5], reverse=True)

        return results

    """
    Performs a similarity search to find the top 'k' most similar documents to the given query embedding.

    Args:
        column_header (str): the name of the column that contains the embeddings you wish to search
        ignore_user (bool): If we want to ignore the user request; this is useful if you want the recall of the vector database to ignore your prompt for the round found in the vector database. This saves tokens by inserting a small user 
            request (i.e. 'Update me.') and then having the full assistant response
        ignore_assistant (bool): If we want to ignore the assistant response; this is useful if you want the recall of the vector database to ignore the assistant's response for the round found in the vector database. This saves tokens by 
            inserting a small assistant response (i.e. 'Fascinating.') and then leaving the associated user request intact. Use this if the assistant is chatty and you want the assistant to fully focus on what YOU said.
        query_embedding (): The query embedding
        k (in): The number of top similar documents to retrieve. 

    Returns:
        List[Tuple[str, str, int, str, int, float]]: A list of tuples, where each tuple contains (column_header, user_request, user_token_count, assistant_response, assistant_token_count, similarity_score) for the top 'k' documents. The list is sorted by similarity score in descending order.
                                 Returns an empty list if the database is empty.
    """
    def _get_search_results(self, column_header, query_embedding, ignore_user: bool, ignore_assistant: bool, top_k):
        # Apply the cosine similarity function to each embedding in the DataFrame
        # The .apply() method iterates through each row's 'embedding' and calculates similarity.
        similarities = self.df[column_header].apply(
            lambda doc_embedding: self._cosine_similarity(query_embedding, doc_embedding)
        )

        temp_generic_user_tokens = self.find_token_count("user", self.USER_REQUEST)
        temp_generic_assistant_tokens = self.find_token_count("assistant", self.ASSISTANT_RESPONSE)

        # Get the indices of the top 'k' largest similarity scores.
        # nlargest(k) returns a Series with the k largest values.
        top_k_series = similarities.nlargest(top_k)

        # Retrieve the actual texts and their corresponding scores using the indices.
        results = []
        for idx, score in top_k_series.items():
            # if we are ignoring the user, write some generic line and get its token count
            if ignore_user:
                user_request = self.USER_REQUEST
                user_token_count = temp_generic_user_tokens
            else:
                user_request = self.df.loc[idx, 'user_text'] # .loc is used for label-based indexing
                user_token_count = self.df.loc[idx, 'user_token_count'] # .loc is used for label-based indexing

            # if we are ignoring the assistants response, write some generic line and get its token count
            if ignore_assistant:
                assistant_response = self.ASSISTANT_RESPONSE
                assistant_token_count = temp_generic_assistant_tokens
            else:
                assistant_response = self.df.loc[idx, 'assistant_text'] # .loc is used for label-based indexing
                assistant_token_count = self.df.loc[idx, 'assistant_token_count'] # .loc is used for label-based indexing


            results.append((column_header, user_request, user_token_count, assistant_response, assistant_token_count, score))

        return results


    def find_token_count(self, role: str, prompt: str) -> int:
        if role not in ['system', 'user', 'assistant']:
            raise ValueError(f"Unsupported role: {role}. Expected 'system', 'user', or 'assistant'.")

        with self.generating_gpu_lock:
            return LlamaUtils.universal_token_count(self.llm_generator, role, prompt, self.generating_model_type)

    def save(self, file_path: str):
        """
        Saves the DataFrame containing documents and embeddings to a Parquet file.

        Args:
            file_path (str): The full path to the Parquet file.
        """
        try:
            if self.passphrase:
                buffer = io.BytesIO()
                self.df.to_parquet(buffer, index=False)
                self.encryption.encrypt_and_save(buffer, file_path)
            else:
                # Ensure the directory exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                self.df.to_parquet(file_path, index=False)
            if self.debug: print(f"{ColoredText.CYAN_TEXT}CppVectorDB.save: Vector database saved to {file_path}.{ColoredText.END_TEXT}")
        except Exception as e:
            print(f"{ColoredText.RED_TEXT}CppVectorDB.save: Error saving vector database: {e}.{ColoredText.END_TEXT}")

    def load(self, file_path: str) -> bool:
        """
        Loads the DataFrame containing documents and embeddings from a Parquet file.

        Args:
            file_path (str): The full path to the Parquet file.

        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        if not os.path.exists(file_path):
            if self.debug: print(f"{ColoredText.CYAN_TEXT}VectorDB.load: Vector database file not found at {file_path}. Starting with empty DB.{ColoredText.END_TEXT}")

            self.df = VectorDB.get_base_dataframe()
            self._expected_embedding_dim = None # Reset dimension expectation if starting empty
            return False
        try:
            if self.passphrase:
                buffer = self.encryption.load_and_decrypt(file_path, return_type='buffer')
                self.df = pd.read_parquet(buffer)
            else:
                self.df = pd.read_parquet(file_path)
            # Ensure 'embedding' column is treated as object dtype after loading
            self.df['embedding'] = self.df['embedding'].astype(object)
            self.df['user_embedding'] = self.df['user_embedding'].astype(object)
            self.df['assistant_embedding'] = self.df['assistant_embedding'].astype(object)
            
            # After loading, try to infer and set _expected_embedding_dim
            if not self.df.empty and 'embedding' in self.df.columns and isinstance(self.df['embedding'].iloc[0], np.ndarray):
                self._expected_embedding_dim = self.df['embedding'].iloc[0].shape[0]
                if self.debug: print(f"{ColoredText.CYAN_TEXT}CppVectorDB.load: Inferred expected embedding dimension from loaded DB: {self._expected_embedding_dim}.{ColoredText.END_TEXT}")
            else:
                self._expected_embedding_dim = None
                if self.debug: print(f"{ColoredText.CYAN_TEXT}CppVectorDB.load: Could not infer expected embedding dimension from loaded DB (it might be empty or corrupted).{ColoredText.END_TEXT}")

            if self.debug: print(f"{ColoredText.CYAN_TEXT}CppVectorDB.load: Vector database loaded from {file_path}.{ColoredText.END_TEXT}")
            return True
        except Exception as e:
            print(f"{ColoredText.RED_TEXT}VectorDB.load: Error loading vector database: {e}. Starting with empty DB.{ColoredText.END_TEXT}")
            self.df = VectorDB.get_base_dataframe()
            self._expected_embedding_dim = None # Reset dimension expectation on error
            return False


    """
    This prepares a new vector database with an empty pandas DataFrame. The DataFrame will store the columns:
        - 'embedding':                  The numpy array representing the embedding of the _combined_ text (user request AND AI assistant response). The 'embedding' column is explicitly set to 'object' dtype
                                        to correctly store numpy arrays. 
                                        
        - 'user_text':                  The user 'prompt' - i.e. your question / comment to the AI assistant.
        
        - 'user_embedding':             The numpy array representing the embedding of the user request / prompt. The 'embedding' column is explicitly set to 'object' dtype to correctly store numpy arrays.
        
        - 'user_token_count':           The token count that will (likely) represent the token count of this string as read by the generator LLM; this is a part of the number that cannot exceed n_ctx.
                                        This is just an approximation, as its hard to get the real count without running your _entire_ chat history through at once - the LLM will probably re-use some tokens too, 
                                        so the token count may be less than this for this user entry.
                                        
        - 'assistant_text':             The AI assistant 'response' to the user prompt.
        
        - 'assistant_embedding':        The numpy array representing the embedding of the AI assistant response. The 'embedding' column is explicitly set to 'object' dtype to correctly store numpy arrays.
        
        - 'assistant_token_count':      The token count that will (likely) represent the token count of this string as read by the generator LLM; this is a part of the number that cannot exceed n_ctx.
                                        This is just an approximation, as its hard to get the real count without running your _entire_ chat history through at once - the LLM will probably re-use some tokens too, 
                                        so the token count may be less than this for this user entry.
    """
    @staticmethod
    def get_base_dataframe():
        #self.df = pd.DataFrame(columns=['text', 'embedding', 'token_count'])
        new_df = pd.DataFrame(columns=['embedding', 'user_text', 'user_embedding', 'user_token_count', 'assistant_text', 'assistant_embedding', 'assistant_token_count'])

        # Explicitly set the dtype for the 'embedding' column to object to ensure it can store numpy arrays correctly.
        new_df['embedding'] = new_df['embedding'].astype(object)
        new_df['user_embedding'] = new_df['user_embedding'].astype(object)
        new_df['assistant_embedding'] = new_df['assistant_embedding'].astype(object)

        return new_df

    def save_session(self, history: List[dict]):
        """Saves the current session data."""
        os.makedirs(self.save_dir, exist_ok=True)
        self.save(self.vector_db_file)
        try:
            if self.passphrase:
                json_str = json.dumps(history, indent=4, ensure_ascii=False)
                self.encryption.encrypt_and_save(json_str, self.chat_history_file)
            else:
                with open(self.chat_history_file, 'w', encoding='utf-8') as f:
                    json.dump(history, f, indent=4)
            if self.debug: print(f"{ColoredText.CYAN_TEXT}VectorDB.save_session: Chat history saved to {self.chat_history_file}.{ColoredText.END_TEXT}")
        except Exception as e:
            print(f"{ColoredText.RED_TEXT}VectorDB.save_session: Error saving chat history: {e}.{ColoredText.END_TEXT}")

    """

    '-> List[dict]:' simply means 'expect this function to return a list of dictionaries'
    """
    def load_session(self) -> List[dict]:
        """Loads a previous session data."""
        loaded_history = []
        if not os.path.exists(self.save_dir):
            if self.debug: print(f"{ColoredText.CYAN_TEXT}VectorDB.load_session: Session directory '{self.save_dir}' not found. No previous session to load.{ColoredText.END_TEXT}")
            return []

        # Attempt to load vector database
        db_loaded = self.load(self.vector_db_file)

        # Attempt to load chat history
        if os.path.exists(self.chat_history_file):
            try:
                if self.passphrase:
                    # Decrypt and load
                    json_str = self.encryption.load_and_decrypt(self.chat_history_file, return_type='str')
                    loaded_history = json.loads(json_str)
                else:
                    with open(self.chat_history_file, 'r', encoding='utf-8') as f:
                        loaded_history = json.load(f)
                if self.debug: print(f"{ColoredText.CYAN_TEXT}VectorDB.load_session: Chat history loaded from {self.chat_history_file}.{ColoredText.END_TEXT}")
            except json.JSONDecodeError as e:
                print(f"{ColoredText.RED_TEXT}VectorDB.load_session: Error decoding chat history JSON: {e}. Starting with empty history.{ColoredText.END_TEXT}")
                loaded_history = []
            except Exception as e:
                print(f"{ColoredText.RED_TEXT}VectorDB.load_session: Error loading chat history: {e}. Starting with empty history.{ColoredText.END_TEXT}")
                loaded_history = []
        else:
            print(f"{ColoredText.CYAN_TEXT}VectorDB.load_session: Chat history file not found at {self.chat_history_file}. Starting with empty history.{ColoredText.END_TEXT}")

        if not db_loaded and not loaded_history:
            print(f"{ColoredText.CYAN_TEXT}VectorDB.load_session: No complete session data loaded. Starting fresh.{ColoredText.END_TEXT}")
        return loaded_history

    def validate_passphrase(self) -> bool:
        """
        Validates the passphrase by attempting to decrypt existing files.
        Returns True if passphrase is correct or no encrypted files exist.
        """
        if not self.passphrase:
            return True  # No encryption, always valid

        # Check if any encrypted files exist
        files_to_check = []
        if os.path.exists(self.vector_db_file):
            files_to_check.append(self.vector_db_file)
        if os.path.exists(self.chat_history_file):
            files_to_check.append(self.chat_history_file)

        if not files_to_check:
            return True  # No files to validate against

        # Try decrypting a small portion of the first file
        try:
            with open(files_to_check[0], 'rb') as f:
                encrypted_sample = f.read()
            self.encryption.cipher.decrypt(encrypted_sample)
            return True
        except:
            return False

    """
    Simply prints the dataframe
    """
    def printDF(self):
        print(self.df)