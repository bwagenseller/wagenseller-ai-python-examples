"""
File Encryption/Decryption Utility using AmadeoEncryption

Usage:
    python amadeo_encryption.py encrypt <input_file> [output_file] [--salt <salt_file>]
    python amadeo_encryption.py decrypt <input_file> [output_file] [--salt <salt_file>]

If output_file is not specified, adds/removes .encrypted extension.
If --salt is not specified, uses 'amadeo.salt' in the same directory as input_file.

Usage examples:

# Default behavior (salt in same dir as input)
python amadeo_encryption.py encrypt document.pdf

# Custom output and default salt
python amadeo_encryption.py encrypt document.pdf encrypted_doc.bin

# Custom salt file
python amadeo_encryption.py encrypt document.pdf --salt /secure/location/my.salt

# Custom output AND custom salt
python amadeo_encryption.py encrypt document.pdf encrypted_doc.bin --salt /secure/my.salt

# Decrypt with custom salt
python amadeo_encryption.py decrypt encrypted_doc.bin document.pdf --salt /secure/my.salt

# Short flag
python amadeo_encryption.py encrypt doc.pdf -s /path/to.salt
"""

import sys
import os
import getpass
from amadeo_utils.misc_utils.FileEncryption import AmadeoEncryption


def get_passphrase(confirm: bool = False) -> str:
    """
    Securely prompt for passphrase.
    
    Args:
        confirm: If True, ask for passphrase twice for confirmation
    """
    passphrase = getpass.getpass("🔑 Enter passphrase: ")
    
    if confirm:
        passphrase2 = getpass.getpass("🔑 Confirm passphrase: ")
        if passphrase != passphrase2:
            print("❌ Passphrases don't match!")
            sys.exit(1)
    
    if len(passphrase) < 8:
        print("⚠️  Warning: Passphrase is short. Recommend at least 16 characters.")
    
    return passphrase


def parse_args():
    """Parse command line arguments."""
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1].lower()
    input_file = sys.argv[2]
    
    # Parse remaining arguments
    output_file = None
    salt_path = None
    
    i = 3
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg == '--salt' or arg == '-s':
            if i + 1 >= len(sys.argv):
                print("❌ --salt requires a path argument")
                sys.exit(1)
            salt_path = sys.argv[i + 1]
            i += 2
        elif output_file is None:
            # First non-flag argument is output file
            output_file = arg
            i += 1
        else:
            print(f"❌ Unexpected argument: {arg}")
            sys.exit(1)
    
    return command, input_file, output_file, salt_path


def main():
    """Main CLI interface."""
    
    # Parse arguments
    command, input_file, output_file, salt_path = parse_args()
    
    if command not in ['encrypt', 'decrypt', 'e', 'd']:
        print("❌ Invalid command. Use 'encrypt' or 'decrypt'")
        sys.exit(1)
    
    # Normalize command
    if command == 'e':
        command = 'encrypt'
    elif command == 'd':
        command = 'decrypt'
    
    # Check input file exists
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    # Determine salt file location if not specified
    if salt_path is None:
        input_dir = os.path.dirname(os.path.abspath(input_file))
        if not input_dir:
            input_dir = '.'
        salt_path = os.path.join(input_dir, 'amadeo.salt')
    
    # For decryption, check if salt file exists
    if command == 'decrypt' and not os.path.exists(salt_path):
        print(f"❌ Salt file not found: {salt_path}")
        print("   Decryption requires the same salt file used during encryption.")
        sys.exit(1)
    
    # Show salt file info
    if os.path.exists(salt_path):
        print(f"📋 Using salt file: {salt_path}")
    else:
        print(f"📋 Will create salt file: {salt_path}")
    
    # Get passphrase (with confirmation for encryption)
    print()
    passphrase = get_passphrase(confirm=(command == 'encrypt'))
    print()
    
    try:
        # Initialize encryption
        print("⏳ Initializing encryption...")
        crypto = AmadeoEncryption(passphrase, salt_path)
        print("✓ Encryption initialized")
        print()
        
        # Execute command
        if command == 'encrypt':
            print(f"📂 Reading: {input_file}")
            file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
            print(f"📊 File size: {file_size_mb:.2f} MB")
            print("🔒 Encrypting...")
            
            result = crypto.encrypt_file(input_file, output_file)
            print(f"✅ Successfully encrypted to: {result}")
        else:  # decrypt
            print(f"📂 Reading: {input_file}")
            print("🔓 Decrypting...")
            
            result = crypto.decrypt_file(input_file, output_file)
            print(f"✅ Successfully decrypted to: {result}")
        
        print("\n✨ Done!")
        
    except ValueError as e:
        print(f"\n❌ {e}")
        sys.exit(1)
    except Exception as e:
        error_msg = str(e)
        if "InvalidToken" in str(type(e).__name__) or "InvalidToken" in error_msg:
            print(f"\n❌ WRONG PASSPHRASE or corrupted file!")
        else:
            print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
