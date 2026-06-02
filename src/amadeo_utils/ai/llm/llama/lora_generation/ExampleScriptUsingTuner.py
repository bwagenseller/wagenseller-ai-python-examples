# run_finetuning.py
import os
import json
import sys # Import sys to create the dummy data file

# --- Import the LoraFineTuner class ---
# This line imports the LoraFineTuner class from finetuner_module.py
from amadeo_utils.ai.llm.llama.lora_generation import LoraFineTuner

def main():
    """
    Main function to orchestrate the fine-tuning process.
    """
    # --- STEP 0: Prepare your data file (if not already done) ---
    # In a real scenario, this data file would already exist and be prepared.
    # This part is just for making the example runnable.
    dummy_data = [
        {"text": "The quick brown fox jumps over the lazy dog in the domain of forest animals. This is a very interesting fact."},
        {"text": "A swift greyhound leaps gracefully over a slumbering feline in the world of domestic pets. They are very fast."},
        {"text": "Quantum entanglement is a phenomenon where two particles are linked, regardless of distance. It's quite bizarre."},
        {"text": "Classical mechanics describes the motion of macroscopic objects, from projectiles to planets. It's fundamental."},
        {"text": "The cat sat on the mat. The dog barked at the mailman. The bird sang a song. This is sample text."}
    ]
    dummy_data_file = "my_domain_data.jsonl"
    if not os.path.exists(dummy_data_file):
        with open(dummy_data_file, 'w', encoding='utf-8') as f:
            for entry in dummy_data:
                f.write(json.dumps(entry) + '\n')
        print(f"Created dummy data file: {dummy_data_file}")
    else:
        print(f"Using existing data file: {dummy_data_file}")


    # --- STEP 1: Define your fine-tuning parameters ---
    # These parameters can be customized for your specific needs.
    # IMPORTANT: Ensure your chosen model, batch_size, and max_seq_length
    # fit within your GPU's VRAM. For 7B models, 16GB-24GB VRAM is often needed.
    
    # Using a very small model for demonstration purposes (e.g., if you don't have a powerful GPU)
    chosen_model_name = "EleutherAI/pythia-160m" 
    
    # For more realistic fine-tuning (requires more VRAM):
    # chosen_model_name = "mistralai/Mistral-7B-v0.1" 
    # chosen_model_name = "meta-llama/Llama-2-7b-hf" # Requires Hugging Face login and access

    output_directory = "./my_finetuned_output"

    print(f"Configuring fine-tuner for model: {chosen_model_name}")
    finetuner = LoraFineTuner(
        model_name=chosen_model_name,
        data_file_path=dummy_data_file,
        output_dir=output_directory,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        # lora_target_modules=["q_proj", "v_proj"], # Customize if needed, defaults are usually good
        batch_size=2, # Keep this low for limited VRAM
        gradient_accumulation_steps=2, # Accumulate gradients over 2 batches, effective batch size 2*2=4
        learning_rate=2e-4,
        num_train_epochs=1, # Keep epochs low for quick testing
        max_seq_length=256, # Limit sequence length to save VRAM
        quantization_4bit=True,
    )

    # --- STEP 2: Load Model, Tokenizer, Data ---
    # These methods are designed to be called internally by `train()`,
    # but you can call them explicitly if you want to inspect objects before training.
    print("\nLoading model, tokenizer, and data...")
    finetuner._load_model_and_tokenizer()
    finetuner._configure_lora()
    finetuner._load_data()
    print("Model, tokenizer, and data loaded successfully and LoRA configured.")

    # --- STEP 3: Run the Training ---
    try:
        print("\nInitiating training process...")
        finetuner.train()
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nERROR during training: {e}", file=sys.stderr)
        print("Please check your GPU memory, data format, and model compatibility.", file=sys.stderr)
        sys.exit(1)

    # --- STEP 4: (Optional) Merge and Save the Full Model ---
    # This step requires enough VRAM to load the full base model again.
    # If your GPU is VRAM-limited, you might skip this and only use the
    # LoRA adapters with the base model during inference.
    try:
        print("\nAttempting to merge LoRA adapters and save full model...")
        finetuner.merge_and_save_model()
        print("\nModel merging and saving completed successfully!")
    except Exception as e:
        print(f"\nERROR during model merging: {e}", file=sys.stderr)
        print("This often indicates insufficient VRAM to load the full base model for merging.", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main() 
