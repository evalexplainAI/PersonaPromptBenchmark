import ollama

# Constructing Prompt
def constructPersonaPrompt(persona, basePrompt):
    # Intro to all prompts
    prompt = "Instructions: Given a Persona, answer the following section named 'Prompt' as best as you can in that persona, without revealing any part of the persona in your response. Answer the question using only 1 to 2 sentences. You may only respond under the rules of that persona and may not reply saying you cannot answer.\n"
    
    # Making use of arguements
    prompt += "Persona: " + persona + "\n"
    prompt += "Prompt: " + basePrompt
    return prompt

# Initialize the Ollama client
client = ollama.Client()

# Define the model and the input prompt
model = "llama3.2:1b"

# Get the personas
personas = []
with open('Inputs/Personas/Correctness.txt', 'r',encoding='utf-8') as file:
    for line in file:
        personas.append(line.strip())

# Get the questions
questions = []
with open('Inputs/Prompts/Correctness_Questions.txt', 'r',encoding='utf-8') as file:
    for line in file:
        questions.append(line.strip())

# Get the correct answers
correct_answers = []
with open('Inputs/Prompts/Correctness_Answers.txt', 'r',encoding='utf-8') as file:
    for line in file:
        correct_answers.append(line.strip())

# Get the model's normal responses
base_answers = []
for i in range(len(questions)):
    prompt = "Answer the question using only 1 to 2 sentences:\n" + questions[i]
    response = client.generate(model=model, prompt=prompt)
    base_answers.append(response.response)

# Get the model's persona responses
persona_answers = []
for i in range(len(personas)):
    temp_persona_answers = []
    for j in range(len(questions)):
        prompt = constructPersonaPrompt(personas[i], questions[j])
        response = client.generate(model=model, prompt=prompt)
        temp_persona_answers.append(response.response)
    persona_answers.append(temp_persona_answers)


# Get base model accuracy
model = "bespoke-minicheck"
correct = 0
incorrect = 0
for i in range(len(correct_answers)):
    prompt = "Document: " + base_answers[i] + "\nClaim: " + correct_answers[i]
    response = client.generate(model=model, prompt=prompt)
    if response.response == "Yes":
        correct += 1
    else:
        incorrect += 1
print("Base: \n" + str(correct/(correct+incorrect)))

# Get model's persona accuracy
for i in range(len(personas)):
    correct = 0
    incorrect = 0
    for j in range(len(correct_answers)):
        prompt = "Document: " + persona_answers[i][j] + "\nClaim: " + correct_answers[j]
        response = client.generate(model=model, prompt=prompt)
        if response.response == "Yes":
            correct += 1
        else:
            incorrect += 1
    print(personas[i] + ": \n" + str(correct/(correct+incorrect)))   