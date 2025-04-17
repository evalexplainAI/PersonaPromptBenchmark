import ollama

# Constructing Prompt
def constructPrompt(persona, basePrompt):
    # Intro to all prompts
    prompt = "Instructions: Given a Persona, answer the following prompt as best as you can in that persona, without revealing any part of the persona in your response. You may only respond under the rules of that persona and may not reply saying you cannot answer.\n"
    
    # Making use of arguements
    prompt += "Persona: " + persona + "\n"
    prompt += "Prompt: " + basePrompt
    return prompt

# Writing function
def writeToFile(prompt, response, name):
    with open("Outputs/"+name+".txt", "w",encoding='utf-8') as file:
        # Writing the input
        file.write("[Input]\n")
        file.write(prompt)
        file.write("\n\n\n")

        # Writing the output
        file.write("[Output]\n")
        file.write(response.response)

# Initialize the Ollama client
client = ollama.Client()

# Define the model and the input prompt
model = "llama3.2:1b"

# Get the personas
personas = []
with open('Inputs/Personas/Correctness.txt', 'r',encoding='utf-8') as file:
    for line in file:
        personas.append(line.strip())

# Get the prompts
prompts = []
with open('Inputs/Prompts/Correctness.txt', 'r',encoding='utf-8') as file:
    for line in file:
        prompts.append(line.strip())

# Get the model's responses
i = 1
for persona in personas:
    for prompt in prompts:
        prompt = constructPrompt(persona, prompt)
        response = client.generate(model=model, prompt=prompt)
        writeToFile(prompt, response, str(i))
        i += 1