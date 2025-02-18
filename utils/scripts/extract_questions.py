import json

from tqdm import tqdm

def extract_questions(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    questions = []
    for item in tqdm(data):
        question = {
            'id': item['id'],
            'question': item['question']
        }
        questions.append(question)

    with open(output_file, 'w') as f:
        json.dump(questions, f, indent=4)

if __name__ == "__main__":
    input_file = 'backgroundata/modelresults/2d-clean-llavanextvideo7b-sabench-v2.1.json'
    output_file = 'raw_data/questions/questions.json'
    extract_questions(input_file, output_file)