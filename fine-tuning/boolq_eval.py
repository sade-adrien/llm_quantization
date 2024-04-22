from transformers import StoppingCriteria
from datasets import load_dataset
from tqdm import tqdm


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_words):
        self.stop_words = stop_words
    def __call__(self, input_ids, scores, **kwargs):
        last_word = tokenizer.decode(input_ids[0,-1])
        return last_word.lower() in self.stop_words


def run_boolq_eval(model, tokenizer):
    dataset = load_dataset('google/boolq')

    accuracy = 0
    fails = 0
    stopping_criteria = StopOnTokens(stop_words=['True', 'true', 'False', 'false'])

    for sample in tqdm(dataset['validation']):
        prompt = f"""Question: {sample['question']}
        Passage: {sample['passage']}
        Answer format: true/false
        Answer:"""

        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=30, stopping_criteria=[stopping_criteria])
        
        if len(outputs[0, inputs.input_ids.shape[1]:]) >= 30:
            fails += 1

        accuracy += tokenizer.decode(outputs[0,-1]).lower() == str(sample['answer']).lower()

    accuracy /= len(dataset['validation'])

    return accuracy, fails

def main():

    model, tokenizer = load_model_tokenizer()
    accuracy, fails = run_boolq_eval(model, tokenizer)

    return accuracy, fails

if __name__ == '__main__':
    main()