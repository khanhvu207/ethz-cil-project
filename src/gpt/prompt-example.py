import openai
import dotenv

config = dotenv.dotenv_values("/cluster/project/infk/krause/chexin/openai/.env")
openai.api_key = config['OPENAI_API_KEY']


def generate_prompt(tweet):
    return """Tell me whether this tweet's sentiment is positive (1) or negative (0):
    {} Please return a single digit only.""".format(
        tweet
    )

tweet = "I love this movie!"
response = openai.Completion.create(
            model="text-davinci-003",
            prompt=generate_prompt(tweet),
            temperature=0,
        )
print(response.choices[0].text)
