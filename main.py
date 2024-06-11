import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import random

tasks = pd.DataFrame(columns=['description', 'priority'])

try:
    tasks = pd.read_csv('tasks.csv')
except FileNotFoundError:
    pass

def save_tasks():
    tasks.to_csv('tasks.csv', index=False)

def train_model():
    if not tasks['description'].empty:
        vectorizer = CountVectorizer()
        clf = MultinomialNB()
        model = make_pipeline(vectorizer, clf)
        model.fit(tasks['description'], tasks['priority'])
        return model
    return None

model = train_model()

def add_task(description, priority):
    global tasks
    new_task = pd.DataFrame({'description': [description], 'priority': [priority]})
    tasks = pd.concat([tasks, new_task], ignore_index=True)
    save_tasks()
    global model
    model = train_model()

def remove_task(description):
    global tasks
    tasks = tasks[tasks['description'] != description]
    save_tasks()
    global model
    model = train_model()

def list_tasks():
    if tasks.empty:
        print("No tasks available.")
    else:
        print(tasks)

def recommend_task():
    if not tasks.empty and model is not None:
        descriptions = tasks['description'].tolist()
        random_description = random.choice(descriptions)
        predicted_priority = model.predict([random_description])[0]
        print(f"Recommended task: {random_description} - Predicted Priority: {predicted_priority}")
    else:
        print("No tasks available for recommendations.")

while True:
    print("\nTask Management App")
    print("1. Add Task")
    print("2. Remove Task")
    print("3. List Tasks")
    print("4. Recommend Task")
    print("5. Exit")

    choice = input("Select an option: ")

    if choice == "1":
        description = input("Enter task description: ")
        priority = input("Enter task priority (Low/Medium/High): ").capitalize()
        add_task(description, priority)
        print("Task added successfully.")

    elif choice == "2":
        description = input("Enter task description to remove: ")
        remove_task(description)
        print("Task removed successfully.")

    elif choice == "3":
        list_tasks()

    elif choice == "4":
        recommend_task()

    elif choice == "5":
        print("Goodbye!")
        break

    else:
        print("Invalid option. Please select a valid option.")
