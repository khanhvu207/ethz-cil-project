from cil.llm import falcon as falcon
from transformers import AutoTokenizer
import pandas as pd
from typing import Optional

def load_data(path: str, data_limit: Optional[int] = None) -> pd.DataFrame:
    # Load a csv file to pandas dataframe
    df = pd.read_csv(path)
    # Randomize each entry of the dataframe
    df = df.sample(frac=1)
    if data_limit is not None:
        # Get the first data_limit rows of the dataframe
        df = df.head(data_limit)
    print("Randomized dataframe:")
    print(df.head())
    return df

def get_raw_return(df, pipeline, tokenizer, prompt_prefix, prompt_postfix):
    # Get the first sentence of each row
    sentences = df["tweet"].tolist()
    response = []
    for s in sentences:
        response.append(
            falcon.generate_text(pipeline, input_text=prompt_prefix + s + prompt_postfix, tokenizer=tokenizer)
        )
        print(response[-1])
    return response

def get_clean_return(response):
    # For each response, identify whether it is positive or negative
    digit_response = []
    positive_keywords = ["1", "positive", "yes"]
    negative_keywords = ["0", "negative", "no"]
    for r in response:
        # See if any of the keywords appear in r.
        if any(k in r.lower() for k in positive_keywords):
            digit_response.append(1)
        elif any(k in r.lower() for k in negative_keywords):
            digit_response.append(0)
        else:
            digit_response.append(0.5)
    return digit_response 

def calculate_accuracy(digit_response, df):
    label = df["label"].tolist()
    same_entries = sum(a == b for a, b in zip(label, digit_response))
    accuracy = same_entries / len(label)
    return accuracy

def main():
    model_id = "tiiuae/falcon-7b-instruct"
    # model_id = "tiiuae/falcon-7b"
    cache_dir = "/cluster/project/infk/krause/chexin/cache/huggingface"
    data_path = "/cluster/project/infk/krause/chexin/ethz_cil_project/data/cleaned_lemmatized_train.csv"
    prompt_prefix = "Tell me whether this sentence is positive (1) or negative (0):"
    prompt_postfix = ". Answer in 1 or 0 only."
    df = load_data(data_path, data_limit=100)
    model = falcon.load_model(model_id, cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    pipeline = falcon.get_pipeline(tokenizer, model)
    raw_responses = get_raw_return(df, pipeline, tokenizer, prompt_prefix, prompt_postfix)
    digit_responses = get_clean_return(raw_responses)
    print(digit_responses)
    print(df["label"].tolist())
    accuracy = calculate_accuracy(digit_responses, df)
    print(f"Accuracy: {accuracy}")
    

if __name__ == "__main__":
    main()
