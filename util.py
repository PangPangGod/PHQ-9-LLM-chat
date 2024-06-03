import random

def get_random_topic():
    topics = [
        "How do you usually spend your weekends?",
        "What's one of your favorite hobbies?",
        "Can you share a memorable experience from your childhood?",
        "What are your thoughts on the importance of mental health?",
        "Have you read any interesting books recently? (book list : 반지의 제왕)"
    ]
    return random.choice(topics)

def transform_json(data):
    new_data = []
    for i in range(len(data) - 1):
        new_entry = {
            "context": {
                "question": data[i]["response"],
                "current_question": data[i]["context"]["current_question"]
            },
            "response": data[i + 1]["context"]["question"]
        }
        new_data.append(new_entry)
    return new_data