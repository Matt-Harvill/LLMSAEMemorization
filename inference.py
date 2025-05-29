from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", torch_dtype=torch.bfloat16, device_map=device)

alice_in_wonderland_text = open("alice_in_wonderland.txt", "r").read()

# We are going to split the alice_in_wonderland_text into chunks of 500 tokens (offset by 30 tokens)
# We are going to batch the chunks together and generate the next 30 tokens for each chunk
# Then for each chunk we are going to count how many tokens are the same as the original text
# We are going to print the chunk that has the most tokens in common with the original text

# First we need to tokenize before splitting into chunks
tokenized_text = tokenizer.encode(alice_in_wonderland_text, add_special_tokens=False)

# Clip for testing
tokenized_text = tokenized_text[:2000]

CHUNK_OFFSET = 30
CHUNK_LENGTH = 500
BATCH_SIZE = 8

# Split the alice_in_wonderland_text into chunks of 500 tokens (offset by 30 tokens)
chunks = [tokenizer.decode(tokenized_text[i:i+CHUNK_LENGTH]) for i in range(0, len(tokenized_text), CHUNK_OFFSET)]

print(f"Number of chunks: {len(chunks)}")
print(f"Number of total tokens: {len(tokenized_text)}")

input("Press Enter to continue...")


for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i+BATCH_SIZE]

    inputs = tokenizer(batch, return_tensors="pt", add_special_tokens=False).to(device)
    outputs = model.generate(**inputs, max_new_tokens=30)

    for j in range(BATCH_SIZE):
        if j >= len(batch):
            break

        input_text = batch[j]
        output_text = tokenizer.decode(outputs[j], skip_special_tokens=True)

        # Get the diff between the original prompt and the newly generated text
        new_text = output_text[len(input_text):]

        # Print the original prompt in blue and the newly generated text in red
        print("\033[94m" + input_text + "\033[0m" + "\033[91m" + new_text + "\033[0m" + "\n")

    input("Press Enter to continue...")