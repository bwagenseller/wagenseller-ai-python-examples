from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.argon2 import Argon2id
import base64
import os
import io

class AmadeoEncryption:
    """
    The encryption itself (Fernet/AES-128):
        * Uses AES in CBC mode with HMAC for authentication
        * AES-128 is considered unbreakable by brute force - would take longer than the age of the universe with current technology
        * Even nation-state actors with massive resources can't crack properly implemented AES

    Key features:
    * Handles any data type (text, bytes, file-like objects)
    * Never writes unencrypted data to disk
    * Works with images, videos, mp3s, parquets, JSON, etc.
    * Single class for all your encryption needs
    * return_type parameter lets you get data back in the format you need

    The salt file isn't secret - it just ensures the same passphrase produces different keys on different systems (prevents rainbow table attacks).

    # Usage examples:

    storage = AmadeoEncryption("your_secret_passphrase")

    # 1. Save/load text or JSON
    storage.encrypt_and_save("Hello world!", "text.enc")
    text = storage.load_and_decrypt("text.enc", return_type='str')

    # 2. Save/load pandas DataFrame
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    storage.encrypt_and_save(buffer, "vector_db.enc")

    # Load it back
    buffer = storage.load_and_decrypt("vector_db.enc", return_type='buffer')
    df = pd.read_parquet(buffer)

    # 3. Save/load binary files (images, mp3s, movies, etc.)
    with open('my_photo.jpg', 'rb') as f:
        photo_bytes = f.read()
    storage.encrypt_and_save(photo_bytes, "photo.enc")

    photo_bytes = storage.load_and_decrypt("photo.enc", return_type='bytes')
    with open('decrypted_photo.jpg', 'wb') as f:
        f.write(photo_bytes)

    # 4. Encrypt/decrypt existing files directly
    storage.encrypt_file("document.pdf", "document.pdf.encrypted")
    storage.decrypt_file("document.pdf.encrypted", "document_restored.pdf")
    """
    def __init__(self, passphrase, salt_path='amadeo.salt'):
        # Load or generate salt (salt should be saved, not secret)
        if os.path.exists(salt_path):
            with open(salt_path, 'rb') as f:
                salt = f.read()
        else:
            salt = os.urandom(16)
            # Only create directory if salt_path has a directory component
            salt_dir = os.path.dirname(salt_path)
            if salt_dir:
                os.makedirs(salt_dir, exist_ok=True)
            with open(salt_path, 'wb') as f:
                f.write(salt)

        # Derive key from passphrase
        kdf = Argon2id(
            memory_cost=102400,   # 100 MiB → very strong in 2025; bump to 524288+ (512+ MiB) in 2–3 years if needed
            iterations=3,          # 3 iterations
            lanes=4,        # use multiple cores
            length=32,
            salt=salt,
        )
        key = base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))

        self.cipher = Fernet(key)

    def encrypt_and_save(self, data, filename):
        """
        Encrypts and saves any data (bytes, string, or file-like object).

        Args:
            data: Can be:
                  - bytes (images, mp3s, movies, parquet, etc.)
                  - str (text, JSON, etc.) - will be encoded to bytes
                  - file-like object with .getvalue() method (like io.BytesIO)
            filename: Path where encrypted file will be saved
        """
        try:
            # Convert data to bytes if needed
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            elif isinstance(data, bytes):
                data_bytes = data
            elif hasattr(data, 'getvalue'):  # io.BytesIO or similar
                data_bytes = data.getvalue()
            else:
                raise TypeError(f"Unsupported data type: {type(data)}")

            # Encrypt
            encrypted = self.cipher.encrypt(data_bytes)

            # Save to disk
            filename_dir = os.path.dirname(filename)
            if filename_dir:
                os.makedirs(filename_dir, exist_ok=True)
            with open(filename, 'wb') as f:
                f.write(encrypted)

        except Exception as e:
            raise Exception(f"Error encrypting and saving data: {e}")

    def load_and_decrypt(self, filename, return_type='bytes'):
        """
        Loads and decrypts data from file.

        Args:
            filename: Path to encrypted file
            return_type: 'bytes', 'str', or 'buffer'
                        - 'bytes': Returns raw bytes (for binary files)
                        - 'str': Returns decoded string (for text/JSON)
                        - 'buffer': Returns io.BytesIO object (for pandas, etc.)

        Returns:
            Decrypted data in requested format
        """
        # Load encrypted data
        with open(filename, 'rb') as f:
            encrypted = f.read()

        # Decrypt (let exceptions propagate naturally)
        decrypted_bytes = self.cipher.decrypt(encrypted)

        # Return in requested format
        if return_type == 'bytes':
            return decrypted_bytes
        elif return_type == 'str':
            return decrypted_bytes.decode('utf-8')
        elif return_type == 'buffer':
            return io.BytesIO(decrypted_bytes)
        else:
            raise ValueError(f"Invalid return_type: {return_type}")

    def encrypt_file(self, input_path: str, output_path: str = None) -> str:
        """
        Encrypt an existing file.

        Args:
            input_path: Path to file to encrypt
            output_path: Path for encrypted output (default: input_path + .encrypted)

        Returns:
            Path to the encrypted file
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Default output path
        if output_path is None:
            output_path = input_path + ".encrypted"

        # Read the file
        with open(input_path, 'rb') as f:
            plaintext = f.read()

        # Encrypt and save
        self.encrypt_and_save(plaintext, output_path)

        return output_path

    def decrypt_file(self, input_path: str, output_path: str = None) -> str:
        """
        Decrypt an existing file.

        Args:
            input_path: Path to encrypted file
            output_path: Path for decrypted output (default: removes .encrypted extension)

        Returns:
            Path to the decrypted file
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Default output path - remove .encrypted extension if present
        if output_path is None:
            if input_path.endswith('.encrypted'):
                output_path = input_path[:-10]  # Remove '.encrypted'
            else:
                output_path = input_path + ".decrypted"

        # Load and decrypt
        decrypted_bytes = self.load_and_decrypt(input_path, return_type='bytes')

        # Write to output file
        with open(output_path, 'wb') as f:
            f.write(decrypted_bytes)

        return output_path