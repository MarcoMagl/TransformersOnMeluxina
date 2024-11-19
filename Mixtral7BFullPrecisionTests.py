"""
Author: Marco Magliulo
Email: marco.magliulo@lxp.lu
Description: This script performs inference on transformer-based models,
            launched with accelerate to leverage multiple GPUs 
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM
import os 

import logging
logging.basicConfig(level=logging.DEBUG)
torch.cuda.empty_cache()

def get_env_variable(var_name):
    if var_name in os.environ:
        return os.environ[var_name]
    else:
        raise KeyError(f"The environment variable '{var_name}' is not defined.")


# Define the names of the environment variables
env_var_names = ["HUGGINGFACEHUB_API_TOKEN", "HF_HOME"]
# Create a dictionary with the environment variable names as keys and their values from the environment
env_vars = {var: os.environ.get(var) for var in env_var_names}
token=env_vars['HUGGINGFACEHUB_API_TOKEN']
cache_dir=env_vars["HF_HOME"]

 
# load the model (should be in the cache directory)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device_map="auto")
# we also need the tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
# some test_cases I have attempted for fun
test_cases = ["SingleQuestionOnF1", "MultipleQuestionsOnF1", "LegalAdvice", "Continuous Prompting"]
test_case = test_cases[-1] 

if test_case == "SingleQuestionOnF1":
    # Initial context
    messages = [
        {"role": "user", "content": "What is your favourite F1 team?"},
        {"role": "assistant", "content": "Well, I love Williams!"},
        {"role": "user", "content": "Why is the Williams team struggling so much nowadays?"}
    ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to("cuda")
    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    print(decoded[0])
elif test_case == "MultipleQuestionsOnF1":

    # Initial context
    messages = [
        {"role": "user", "content": "What is your favourite F1 team?"},
        {"role": "assistant", "content": "Well, I love Williams!"},
    ]
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to("cuda")
    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)

    # Simulate user entering multiple prompts
    user_inputs = [
        "Why Ferrari was so dominant during the era of Schumacher?",
        "How would you describe the driving style of Nikki Lauda?"
        "What were the most important technical innovation in the world of F1 in 90's?"
    ]

    for system_style in ["a regular F1 fan", "a passionate F1 fan with a lot of mechanical knowledge"]:
        # Process user inputs one by one
        for user_input in user_inputs:
            # Prepare the input as before
            chat = [
                {"role": "system", "content": f"You are {system_style}."},
                {"role": "user", "content": user_input}
            ]
            # Apply the chat template
            formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            print("Formatted chat:\n", formatted_chat)
            # Tokenize the chat (This can be combined with the previous step using tokenize=True)
            inputs = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
            # Move the tokenized inputs to the same device the model is on (GPU/CPU)
            inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
            # Generate text from the model
            outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
            decoded_output = tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
            print("Decoded output:\n", decoded_output)

elif test_case == "LegalAdvice":
    import torch
    # Prepare the input as before
    chat = [
        {"role": "system", "content": "You are a sympathetic lawyer. You work in Europe."},
        {"role": "user", "content": "I bought a used car in Luxembourg in a garage and I live in France. After 3 months I got an issue with the engine. Can I ask the selling garage for a compensation?"}
    ]

    formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    print("Formatted chat:\n", formatted_chat)
    inputs = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
    inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
    print_tokenized_inputs = False 
    if print_tokenized_inputs:
        print("Tokenized inputs:\n", inputs)
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
    print_generated_tokens = False 
    if print_generated_tokens:
        print("Tokennized output:\n", print_generated_tokens)
    decoded_output = tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
    print("Decoded output:\n", decoded_output)
elif test_case == "Continuous Prompting" :
    # Initialize the conversation history
    chat_history = [
        {"role": "system", "content": "You are a sympathetic lawyer. You work in Europe."}
    ]

    # Loop to continue accepting user input
    while True:
        user_input = input("You: ")

        # Check if the user wants to end the conversation
        if user_input.lower() in ["exit", "quit"]:
            print("Ending the conversation.")
            break

        # Add the new user input to the conversation
        chat_history.append({"role": "user", "content": user_input})

        # Format the conversation for the model
        formatted_chat = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
        inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}

        # Generate the model's response
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
        decoded_output = tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)

        # Print and add the model's response to the conversation history
        print("Model:", decoded_output)
        chat_history.append({"role": "assistant", "content": decoded_output})
