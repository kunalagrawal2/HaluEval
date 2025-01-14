import random
import openai
import time
import json
import argparse
import tiktoken

import requests

import pandas as pd




openai.api_key = ''

def get_qa_response(model, question, answer, instruction):
    message = [
        {"role": "system", "content":"You are a huallucination detector. You MUST determine if the provided answer contains hallucination or not for the question based on the world knowledge. The answer you provided MUST be \"Yes\" or \"No\""},
        {"role": "user", "content": instruction +
                                    "\n\n#Question#: " + question +
                                    "\n#Answer#: " + answer +
                                    "\n#Your Judgement#: "} 
    ]
    prompt = instruction + "\n\n#Question#: " + question + "\n#Answer#: " + answer + "\n#Your Judgement#:"
    while True:
        try:
            if model == "gpt-3.5-turbo":
                res = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=message,
                    temperature=0.0,
                )
                response = res['choices'][0]['message']['content']
            elif model == "gpt-4o":
                res = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=message,
                    temperature=0.0,
                )
                response = res['choices'][0]['message']['content']
            elif model == "gpt-4o-mini":
                res = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=message,
                    temperature=0.0,
                )
                response = res['choices'][0]['message']['content']    
            else:
                res = openai.Completion.create(
                    engine=model,
                    prompt=prompt,
                    temperature=0.0
                )
                response = res["choices"][0]['text'].strip()
            break
        except openai.error.RateLimitError:
            print('openai.error.RateLimitError\nRetrying...')
            time.sleep(60)
        except openai.error.ServiceUnavailableError:
            print('openai.error.ServiceUnavailableError\nRetrying...')
            time.sleep(20)
        except openai.error.Timeout:
            print('openai.error.Timeout\nRetrying...')
            time.sleep(20)
        except openai.error.APIError:
            print('openai.error.APIError\nRetrying...')
            time.sleep(20)
        except openai.error.APIConnectionError:
            print('openai.error.APIConnectionError\nRetrying...')
            time.sleep(20)
    
    return response


def majority_vote(judgements):
    counts = {"Yes": judgements.count("Yes"), "No": judgements.count("No")}
    if counts["Yes"] > counts["No"]:
        return "Yes"
    elif counts["No"] > counts["Yes"]:
        return "No"
    else:
        return "No" # shouldn't be possible to tie with 3 models, keeping in case have to deal with more than 3 later


def get_dialogue_response(model, dialog, response, instruction):
    message = [
        {"role": "system", "content": "You are a response judge. You MUST determine if the provided response contains non-factual or hallucinated information. The answer you give MUST be \"Yes\" or \"No\""},
        {"role": "user", "content": instruction +
                                    "\n\n#Dialogue History#: " + dialog +
                                    "\n#Response#: " + response +
                                    "\n#Your Judgement#: "}
    ]
    prompt = instruction + "\n\n#Dialogue History#: " + dialog + "\n#Response#: " + response + "\n#Your Judgement#:"
    while True:
        try:
            if model == "gpt-3.5-turbo":
                res = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=message,
                    temperature=0.0,
                )
                response = res['choices'][0]['message']['content']
            elif model == "gpt-4o":
                res = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=message,
                    temperature=0.0,
                )
                response = res['choices'][0]['message']['content']   
            elif model == "gpt-4o-mini":
                res = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=message,
                    temperature=0.0,
                )
                response = res['choices'][0]['message']['content']   
            else:
                res = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    temperature=0.0
                )
                response = res["choices"][0]['text'].strip()
            break
        except openai.error.RateLimitError:
            print('openai.error.RateLimitError\nRetrying...')
            time.sleep(60)
        except openai.error.ServiceUnavailableError:
            print('openai.error.ServiceUnavailableError\nRetrying...')
            time.sleep(20)
        except openai.error.Timeout:
            print('openai.error.Timeout\nRetrying...')
            time.sleep(20)
        except openai.error.APIError:
            print('openai.error.APIError\nRetrying...')
            time.sleep(20)
        except openai.error.APIConnectionError:
            print('openai.error.APIConnectionError\nRetrying...')
            time.sleep(20)

    return response


def num_tokens_from_message(message, model="davinci"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(message))
    return num_tokens


def truncate_message(prompt1, prompt2, model="davinci"):
    if num_tokens_from_message(prompt1 + prompt2, model) > 2033:
        truncation_length = 2033 - num_tokens_from_message(prompt2)
        while num_tokens_from_message(prompt1) > truncation_length:
            prompt1 = " ".join(prompt1.split()[:-1])
    prompt = prompt1 + prompt2
    return prompt


def get_summarization_response(model, document, summary, instruction):
    message = [
        {"role": "system", "content": "You are a summary judge. You MUST determine if the provided summary contains non-factual or hallucinated information. The answer you give MUST be \"Yes\" or \"No\""},
        {"role": "user", "content": instruction +
                                    "\n\n#Document#: " + document +
                                    "\n#Summary#: " + summary +
                                    "\n#Your Judgement#: "}
    ]
    prompt1 = instruction + "\n\n#Document#: " + document
    prompt2 = "\n#Summary#: " + summary + "\n#Your Judgement#:"
    if model == "davinci":
        prompt = truncate_message(prompt1, prompt2)
    else:
        prompt = prompt1 + prompt2
    while True:
        try:
            if model == "gpt-3.5-turbo":
                res = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=message,
                    temperature=0.0,
                )
                response = res['choices'][0]['message']['content']
            elif model == "gpt-4o":
                res = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=message,
                    temperature=0.0,
                )
                response = res['choices'][0]['message']['content']
            elif model == "gpt-4o-mini":
                res = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=message,
                    temperature=0.0,
                )
                response = res['choices'][0]['message']['content']   
            else:
                res = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    temperature=0.0
                )
                response = res["choices"][0]['text'].strip()
            break
        except openai.error.RateLimitError:
            print('openai.error.RateLimitError\nRetrying...')
            time.sleep(60)
        except openai.error.ServiceUnavailableError:
            print('openai.error.ServiceUnavailableError\nRetrying...')
            time.sleep(20)
        except openai.error.Timeout:
            print('openai.error.Timeout\nRetrying...')
            time.sleep(20)
        except openai.error.APIError:
            print('openai.error.APIError\nRetrying...')
            time.sleep(20)
        except openai.error.APIConnectionError:
            print('openai.error.APIConnectionError\nRetrying...')
            time.sleep(20)

    return response


def sentence_similarity(model_response, answer):
    api_token = ""
    API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
    headers = {"Authorization": f"Bearer {api_token}"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    data = query(
        {
            "inputs": {
                "source_sentence": answer,
                "sentences":[model_response]
            }
        })
    return data


def evaluation_qa_dataset(model, file, instruction, output_path):
    similarity_data = pd.DataFrame(columns=['sentence_similarity', 'hallucination'])
    with open(file, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

        
        correct = 0
        incorrect = 0
        for i in range(len(data)):
            knowledge = data[i]["knowledge"]
            question = data[i]["question"]
            hallucinated_answer = data[i]["hallucinated_answer"]
            right_answer = data[i]["right_answer"]

            if random.random() > 0.5:
                answer = hallucinated_answer
                ground_truth = "Yes"
            else:
                answer = right_answer
                ground_truth = "No"

            ans = get_qa_response(model, question, answer, instruction)
            ans = ans.replace(".", "")

            if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
                gen = {"knowledge": knowledge, "question": question, "answer": answer, "ground_truth": ground_truth, "judgement": "failed!"}
                dump_jsonl(gen, output_path, append=True)
                incorrect += 1
                print('sample {} fails......'.format(i))
                continue
            elif "Yes" in ans:
                if ans != "Yes":
                    ans = "Yes"
                gen = {"knowledge": knowledge, "question": question, "answer": answer, "ground_truth": ground_truth, "judgement": ans}
            elif "No" in ans:
                if ans != "No":
                    ans = "No"
                gen = {"knowledge": knowledge, "question": question, "answer": answer, "ground_truth": ground_truth, "judgement": ans}
            else:
                gen = None
                incorrect += 1


            assert(gen is not None)

            row = {"sentence_similarity": sentence_similarity(ans, right_answer), "hallucination": gen['judgement']}
            similarity_data = similarity_data._append(row, ignore_index=True)

            if ground_truth == ans:
                correct += 1
            else:
                incorrect += 1
            print('sample {} success......'.format(i))
            dump_jsonl(gen, output_path, append=True)

        
        similarity_data.to_csv('sentence_similarity.csv', index=True)
        print('{} correct samples, {} incorrect samples, Accuracy: {}'.format(correct, incorrect, correct/len(data)))


def evaluation_dialogue_dataset(model, file, instruction, output_path):
    with open(file, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

        correct = 0
        incorrect = 0
        for i in range(len(data)):
            knowledge = data[i]["knowledge"]
            dialog = data[i]["dialogue_history"]
            hallucinated_response = data[i]["hallucinated_response"]
            right_response = data[i]["right_response"]

            if random.random() > 0.5:
                response = hallucinated_response
                ground_truth = "Yes"
            else:
                response = right_response
                ground_truth = "No"

            ans = get_dialogue_response(model, dialog, response, instruction)
            ans = ans.replace(".", "")

            if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
                gen = {"knowledge": knowledge, "dialogue_history": dialog, "response": response, "ground_truth": ground_truth, "judgement": "failed!"}
                dump_jsonl(gen, output_path, append=True)
                incorrect += 1
                print('sample {} fails......'.format(i))
                continue
            elif "Yes" in ans:
                if ans != "Yes":
                    ans = "Yes"
                gen = {"knowledge": knowledge, "dialogue_history": dialog, "response": response, "ground_truth": ground_truth, "judgement": ans}
            elif "No" in ans:
                if ans != "No":
                    ans = "No"
                gen = {"knowledge": knowledge, "dialogue_history": dialog, "response": response, "ground_truth": ground_truth, "judgement": ans}
            else:
                gen = None
            assert (gen is not None)

            if ground_truth == ans:
                correct += 1
            else:
                incorrect += 1

            print('sample {} success......'.format(i))
            dump_jsonl(gen, output_path, append=True)

        print('{} correct samples, {} incorrect samples, Accuracy: {}'.format(correct, incorrect, correct / len(data)))


def evaluation_qa_ensemble(models, file, instruction, output_path):
    with open(file, 'r', encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    correct, incorrect = 0, 0

    for i, record in enumerate(data):
        knowledge = record["knowledge"]
        question = record["question"]
        hallucinated_answer = record["hallucinated_answer"]
        right_answer = record["right_answer"]

        if random.random() > 0.5:
            answer = hallucinated_answer
            ground_truth = "Yes"
        else:
            answer = right_answer
            ground_truth = "No"

        # Get responses from all models
        model_judgements = []
        for model in models:
            print(f"Model: {model}")
            response = get_qa_response(model, question, answer, instruction)
            model_judgements.append(response)

        # Ensemble by majority vote
        final_judgement = majority_vote(model_judgements)
        
        judgement = final_judgement

        result = {
            "knowledge": knowledge,
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "judgement": judgement
        }

        # Check correctness
        if ground_truth == final_judgement:
            correct += 1
            print(f'Sample {i} success...')
        else:
            incorrect += 1
            print(f'Sample {i} failed...')

        dump_jsonl(result, output_path, append=True)

    total = correct + incorrect
    accuracy = correct / total if total > 0 else 0
    print(f'{correct} correct samples, {incorrect} incorrect samples, Accuracy: {accuracy:.2f}')


def evaluation_summarization_dataset(model, file, instruction, output_path):
    with open(file, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

        correct = 0
        incorrect = 0
        for i in range(len(data)):

            document = data[i]["document"]
            hallucinated_summary = data[i]["hallucinated_summary"]
            right_summary = data[i]["right_summary"]

            if random.random() > 0.5:
                summary = hallucinated_summary
                ground_truth = "Yes"
            else:
                summary = right_summary
                ground_truth = "No"

            ans = get_summarization_response(model, document, summary, instruction)
            ans = ans.replace(".", "")

            if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
                gen = {"document": document, "summary": summary, "ground_truth": ground_truth, "judgement": "failed!"}
                dump_jsonl(gen, output_path, append=True)
                incorrect += 1
                print('sample {} fails......'.format(i))
                continue
            elif "Yes" in ans:
                if ans != "Yes":
                    ans = "Yes"
                gen = {"document": document, "summary": summary, "ground_truth": ground_truth, "judgement": ans}
            elif "No" in ans:
                if ans != "No":
                    ans = "No"
                gen = {"document": document, "summary": summary, "ground_truth": ground_truth, "judgement": ans}
            else:
                gen = None
            assert (gen is not None)

            if ground_truth == ans:
                correct += 1
            else:
                incorrect += 1

            print('sample {} success......'.format(i))
            dump_jsonl(gen, output_path, append=True)

        print('{} correct samples, {} incorrect samples, Accuracy: {}'.format(correct, incorrect, correct / len(data)))


def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
            json_record = json.dumps(data, ensure_ascii=False)
            f.write(json_record + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hallucination Generation")

    parser.add_argument("--task", default="qa", help="qa, dialogue, or summarization")
    #parser.add_argument("--model", default="davinci", help="model name")
    parser.add_argument("--models", nargs='+', help="models to use for ensembling")
    args = parser.parse_args()

    instruction_file = "{}/{}_evaluation_instruction.txt".format(args.task, args.task)
    f = open(instruction_file, 'r', encoding="utf-8")
    instruction = f.read()

    #model = args.model
    models = args.models
    model = models[0]
    output_path = "{}/{}_{}_results.json".format(args.task, args.task, model)

    data = "../data/{}_data.json".format(args.task)

    if args.task == "qa":
        if len(models) == 1:
            evaluation_qa_dataset(model, data, instruction, output_path)
        else:
            output_path = "{}/{}_{}_results.json".format(args.task, args.task, "ensemble")
            evaluation_qa_ensemble(models, data, instruction, output_path) 
    elif args.task == "dialogue":
        evaluation_dialogue_dataset(model, data, instruction, output_path)
    elif args.task == "summarization":
        evaluation_summarization_dataset(model, data, instruction, output_path)
    else:
        raise ValueError("The task must be qa, dialogue, or summarization!")
