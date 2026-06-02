import os
import torch
import json # Used for loading JSONL data

# Import necessary classes from Hugging Face ecosystem
from transformers import (
    AutoModelForCausalLM, # For loading pre-trained causal language models
    AutoTokenizer,        # For loading the model's tokenizer
    BitsAndBytesConfig,   # For 4-bit quantization (memory efficiency)
    TrainingArguments,    # For defining training parameters
)
from peft import (
    LoraConfig,           # For configuring LoRA parameters
    get_peft_model,       # To apply LoRA to the base model
    PeftModel            # To load and merge PEFT (LoRA) models
)
from trl import SFTTrainer    # A trainer specifically for Supervised Fine-Tuning
from datasets import load_dataset # For loading datasets efficiently

# --- IMPORTANT PREREQUISITES ---
# Ensure you have these libraries installed:
# pip install torch transformers peft trl bitsandbytes accelerate datasets


class LoraFineTuner:
    """
    A class to facilitate LoRA (Low-Rank Adaptation) fine-tuning of Large Language Models.

    This class handles data loading, model quantization, LoRA configuration,
    training execution, and saving of LoRA adapters, and optionally merging
    them back into the base model.
    """

    def __init__(self,
                 model_name: str,
                 data_file_path: str,
                 output_dir: str,
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.05,
                 lora_target_modules: list = None,
                 batch_size: int = 4,
                 gradient_accumulation_steps: int = 4,
                 learning_rate: float = 2e-4,
                 num_train_epochs: int = 3,
                 max_seq_length: int = 1024,
                 quantization_4bit: bool = True):
        """
        Initializes the LoRA Fine-Tuner with model, data, and training configurations.

        Args:
            model_name (str): The name or path of the base pre-trained LLM from Hugging Face.
                              E.g., "mistralai/Mistral-7B-v0.1", "meta-llama/Llama-2-7b-hf".
            data_file_path (str): Path to your domain-specific dataset (must be in JSONL format).
                                  Each line should be a JSON object with a 'text' key,
                                  or 'prompt' and 'completion' keys depending on your data formatting.
            output_dir (str): Directory where the trained LoRA adapters and/or merged model will be saved.
            lora_r (int): LoRA attention dimension (rank). Higher 'r' means more expressivity but more parameters.
            lora_alpha (int): LoRA scaling factor. Controls the magnitude of updates.
            lora_dropout (float): Dropout probability for LoRA layers.
            lora_target_modules (list, optional): List of module names to apply LoRA to.
                                                Defaults to common attention layers if None.
            batch_size (int): Training batch size per device.
            gradient_accumulation_steps (int): Number of updates steps to accumulate gradients before performing a backward/update pass.
            learning_rate (float): Initial learning rate for the optimizer.
            num_train_epochs (int): Total number of training epochs to perform.
            max_seq_length (int): Maximum sequence length for tokenization. Important for GPU memory.
            quantization_4bit (bool): Whether to load the base model in 4-bit precision (reduces VRAM usage).
        """
        print(f"Initializing LoRA Fine-Tuner for model: {model_name}")

        self.model_name = model_name
        self.data_file_path = data_file_path
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # --- 4-bit Quantization Configuration (for memory efficiency) ---
        self.bnb_config = None
        if quantization_4bit:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,                 # Load model weights in 4-bit
                bnb_4bit_quant_type="nf4",         # Quantization type: NormalFloat4 is generally good
                bnb_4bit_compute_dtype=torch.bfloat16, # Compute in bfloat16 for speed and precision
                bnb_4bit_use_double_quant=False,   # Whether to use double quantization (more memory savings, slight speed penalty)
            )
            print("4-bit quantization enabled.")
        else:
            print("4-bit quantization disabled. Model will load in full precision.")

        # --- LoRA Configuration ---
        # lora_target_modules depend on the model architecture.
        # Common choices include 'q_proj', 'v_proj', 'k_proj', 'o_proj' for attention layers.
        # Some models also use 'gate_proj', 'up_proj', 'down_proj' for MLP layers.
        if lora_target_modules is None:
            # Default target modules common for Llama/Mistral architectures
            self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            print(f"Using default LoRA target modules: {self.lora_target_modules}")
        else:
            self.lora_target_modules = lora_target_modules
            print(f"Using custom LoRA target modules: {self.lora_target_modules}")

        self.peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",        # Bias can be "none", "all", or "lora_only"
            task_type="CAUSAL_LM", # This specifies that we're fine-tuning a language model
            target_modules=self.lora_target_modules,
        )
        print(f"LoRA configuration: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")

        # --- Training Arguments ---
        self.training_args = TrainingArguments(
            output_dir=self.output_dir,                # Directory to save logs and checkpoints
            per_device_train_batch_size=batch_size,    # Batch size per GPU/CPU
            gradient_accumulation_steps=gradient_accumulation_steps, # Accumulate gradients over steps
            learning_rate=learning_rate,               # Learning rate for optimizer
            num_train_epochs=num_train_epochs,         # Number of passes over the dataset
            optim="paged_adamw_8bit",                  # Optimizer (paged for memory efficiency)
            logging_steps=25,                          # Log every N steps
            save_strategy="epoch",                     # Save model at the end of each epoch
            fp16=True if torch.cuda.is_available() else False, # Use mixed precision if GPU available
            report_to="none",                          # Disable reporting to platforms like wandb or tensorboard
            # You might add: evaluation_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="eval_loss"
            # if you have a validation set.
        )
        print(f"Training arguments: Batch Size={batch_size}, Epochs={num_train_epochs}, LR={learning_rate}")

        # Initialize model and tokenizer to None; loaded later
        self.model = None
        self.tokenizer = None
        self.dataset = None

    def _load_data(self):
        """
        Loads the dataset from the specified JSONL file.
        The dataset is expected to be in a format compatible with SFTTrainer,
        e.g., each row is a JSON object with a 'text' field, or 'prompt'/'completion'.
        """
        print(f"Loading dataset from: {self.data_file_path}")
        # The 'datasets' library can load JSONL files directly
        # 'split="train"' loads the entire dataset as the training split
        self.dataset = load_dataset("json", data_files=self.data_file_path, split="train")
        print(f"Dataset loaded. Number of examples: {len(self.dataset)}")
        # Example of how to see one example:
        # print("First dataset example:", self.dataset[0])

        # If your data is in a format like {"prompt": "...", "completion": "..."},
        # SFTTrainer will automatically concatenate them.
        # If it's just {"text": "..."}, make sure it's already formatted for training (e.g., instructions).
        # You can add a formatting function here if needed, like in the conceptual outline.
        # For simplicity, we assume 'text' field is ready for direct training.

    def _load_model_and_tokenizer(self):
        """
        Loads the base pre-trained LLM and its tokenizer.
        Applies 4-bit quantization if enabled during initialization.
        """
        print(f"Loading base model '{self.model_name}'...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config, # Applies 4-bit quantization if config is set
            device_map="auto",                   # Automatically distributes model across available devices (GPUs)
            torch_dtype=torch.bfloat16 if self.bnb_config and self.bnb_config.bnb_4bit_compute_dtype == torch.bfloat16 else torch.float16 # Use bfloat16 for compute if specified, else float16
        )
        # Disable cache for more memory efficiency during training
        self.model.config.use_cache = False
        # Set pretraining_tp to 1 for single-GPU setups (helps avoid some warnings)
        self.model.config.pretraining_tp = 1

        print(f"Loading tokenizer for '{self.model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        # Set pad_token and padding_side, crucial for batching and consistent input lengths
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right" # Models like Llama/Mistral prefer right padding

        print("Model and tokenizer loaded successfully.")

    def _configure_lora(self):
        """
        Applies the LoRA configuration to the loaded base model.
        """
        print("Applying LoRA configuration to the model...")
        self.model = get_peft_model(self.model, self.peft_config)
        # Print a summary of trainable parameters (only LoRA adapters are trainable)
        self.model.print_trainable_parameters()
        print("LoRA configured.")

    def train(self):
        """
        Executes the LoRA fine-tuning process.
        """
        if not all([self.model, self.tokenizer, self.dataset]):
            print("Pre-requisites for training (model, tokenizer, dataset) not loaded. Loading them now.")
            self._load_model_and_tokenizer()
            self._configure_lora() # Apply LoRA after loading model
            self._load_data()

        print("\n--- Starting LoRA Fine-Tuning ---")

        # Initialize the SFTTrainer
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset,
            tokenizer=self.tokenizer,
            args=self.training_args,
            packing=False, # Set to True if you want to pack multiple short examples into one sequence
                           # (can improve GPU utilization, but requires specific data format for 'text' field)
            max_seq_length=self.max_seq_length,
            # If your dataset has separate 'prompt' and 'completion' fields,
            # you might need to provide a formatting function here or preprocess the dataset
            # before passing it to SFTTrainer.
        )

        # Start the training!
        trainer.train()
        print("\n--- LoRA Fine-Tuning Complete! ---")

        # Save the LoRA adapters
        self.save_lora_adapters(trainer)
        
    def save_lora_adapters(self, trainer):
        """
        Saves only the trained LoRA adapters.
        These are small files containing the delta weights.
        """
        lora_save_path = os.path.join(self.output_dir, "lora_adapters")
        print(f"Saving LoRA adapters to: {lora_save_path}")
        trainer.save_model(lora_save_path)
        self.tokenizer.save_pretrained(lora_save_path) # It's good practice to save tokenizer too
        print("LoRA adapters saved.")
        print("Check the contents: You'll typically find `adapter_model.bin` (or `model.safetensors` in newer PEFT versions) and `adapter_config.json`.")

    def merge_and_save_model(self):
        """
        Merges the trained LoRA adapters back into the original base model
        and saves the complete, fine-tuned model (including original and adapted weights).
        This merged model can then be used directly for inference without needing PEFT.
        """
        print("\n--- Merging LoRA adapters into base model ---")
        merged_model_path = os.path.join(self.output_dir, "merged_model")
        os.makedirs(merged_model_path, exist_ok=True)

        # 1. Load the original base model (in full precision if possible, or bfloat16)
        print(f"Loading base model '{self.model_name}' for merging...")
        # Ensure the base model is loaded in the correct dtype for merging.
        # Often float16 or bfloat16 is preferred for merged models.
        base_model_for_merge = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            return_dict=True, # Ensure it returns a dictionary for easy use with PeftModel
            torch_dtype=torch.float16, # Or torch.bfloat16 if your GPU supports it and model prefers
            device_map="cpu", # Load on CPU first to avoid VRAM issues before merging
                              # then transfer to GPU if needed for inference
        )
        print("Base model loaded for merging.")

        # 2. Load the PEFT model (LoRA adapters)
        lora_adapters_path = os.path.join(self.output_dir, "lora_adapters")
        print(f"Loading LoRA adapters from: {lora_adapters_path}")
        model_to_merge = PeftModel.from_pretrained(base_model_for_merge, lora_adapters_path)
        print("LoRA adapters loaded.")

        # 3. Merge the adapters into the base model
        print("Merging LoRA adapters. This might take a moment...")
        merged_model = model_to_merge.merge_and_unload() # This performs the merge operation
        print("LoRA adapters merged.")

        # 4. Save the merged model and tokenizer
        print(f"Saving merged model to: {merged_model_path}")
        merged_model.save_pretrained(merged_model_path, safe_serialization=True) # safe_serialization=True saves as .safetensors
        self.tokenizer.save_pretrained(merged_model_path)
        print(f"Merged model and tokenizer saved to {merged_model_path}.")
        print("The model weights will typically be saved as `model.safetensors`.")

# --- Example Usage ---
if __name__ == '__main__':
    # --- STEP 0: Prepare your data file ---
    # Create a dummy data.jsonl for demonstration.
    # In a real scenario, this would be your domain-specific text.
    # Each line is a JSON object. For SFTTrainer, a "text" key is common.
    # Make sure to format it appropriately for your desired fine-tuning task.
    dummy_data = [
        {"text": "The quick brown fox jumps over the lazy dog in the domain of forest animals."},
        {"text": "A swift greyhound leaps gracefully over a slumbering feline in the world of domestic pets."},
        {"text": "Quantum entanglement is a phenomenon where two particles are linked, regardless of distance."},
        {"text": "Classical mechanics describes the motion of macroscopic objects, from projectiles to planets."},
    ]
    dummy_data_file = "domain_data.jsonl"
    with open(dummy_data_file, 'w', encoding='utf-8') as f:
        for entry in dummy_data:
            f.write(json.dumps(entry) + '\n')
    print(f"Created dummy data file: {dummy_data_file}")


    # --- STEP 1: Define your fine-tuning parameters ---
    # Choose a small, instruction-tuned model for faster demonstration.
    # For real use, consider Mistral-7B, Llama-2-7b, or Gemma-2B/7B.
    # Make sure you have enough VRAM for your chosen model size and max_seq_length!
    # For a 7B model, 24GB VRAM or more is usually recommended even with 4-bit LoRA.
    # For testing, you might use a tiny model like "EleutherAI/pythia-160m" if VRAM is very limited.
    
    # Using a very small model for testing if you have limited VRAM
    # For a more useful model, pick a 7B parameter model (requires >12GB VRAM usually)
    chosen_model_name = "EleutherAI/pythia-160m" 
    # Or for a more serious try (requires ~16GB-24GB VRAM):
    # chosen_model_name = "mistralai/Mistral-7B-v0.1" 
    # chosen_model_name = "meta-llama/Llama-2-7b-hf" # Requires Hugging Face login and access

    finetuner = LoraFineTuner(
        model_name=chosen_model_name,
        data_file_path=dummy_data_file,
        output_dir="./fine_tuned_llama_cpp_model", # Output directory for saved LoRA/merged model
        lora_r=8,
        lora_alpha=16,
        num_train_epochs=1, # Keep epochs low for quick testing
        batch_size=2,
        gradient_accumulation_steps=2,
        max_seq_length=256, # Adjust based on data length and VRAM
        quantization_4bit=True,
    )

    # --- STEP 2: Load Model, Tokenizer, Data ---
    finetuner._load_model_and_tokenizer() # Load the base model and tokenizer
    finetuner._configure_lora()           # Apply LoRA config
    finetuner._load_data()                # Load your dataset

    # --- STEP 3: Run the Training ---
    try:
        finetuner.train()
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        print("Common reasons: Insufficient GPU VRAM, incorrect data format, network issues.")
        print("Try reducing batch_size, gradient_accumulation_steps, or max_seq_length. Or choose a smaller base model.")
        sys.exit(1)

    # --- STEP 4: (Optional) Merge and Save the Full Model ---
    # This step is resource-intensive as it loads the full base model again.
    # Only run this if you have enough VRAM for the full base model.
    # If your GPU VRAM is limited, you might skip this and just use the LoRA adapters
    # with the base model in an inference framework that supports it.
    try:
        finetuner.merge_and_save_model()
    except Exception as e:
        print(f"\nAn error occurred during merging and saving the model: {e}")
        print("This usually happens if you don't have enough VRAM to load the full base model for merging.")
        print("You can still use the LoRA adapters directly with the base model during inference if your framework supports it.")
        sys.exit(1)
        
    print("\nFine-tuning process completed successfully!") 
