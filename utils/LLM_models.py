
import openai

def LLMs(model_name,prompt):
    """
    :param model_name:
    :param prompt:
    :return:
    """
    openai.api_key = "EMPTY"
    openai.api_base = "http://localhost:8000/v1"
    completion = openai.ChatCompletion.create(
    model=model_name,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=512,
    timeout=1800,
    )
    answer = completion.choices[0].message.content
    return answer

if __name__ == '__main__':
    pass