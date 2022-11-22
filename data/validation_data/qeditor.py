import keyboard
import pickle
with open("questions", "rb") as fp:
    questions = pickle.load(fp)
try:
    with open("remaining", "rb") as fp2:
        remaining = pickle.load(fp2)
except:
    remaining = []
interrupt = False
for idx, question in enumerate(questions):
    try:
        print(question)
        command = input()
        if command == "d":
            continue
        elif command == "s":
            remaining.append(question)
        elif command == "m":
            modified = input("Enter new modified sentence:\n")
            remaining.append(modified)
    except:
        print(f"Canceled at question {idx}")
        interrupt = True
        break
if interrupt:
    questions = questions[idx:]
else:
    questions = []
with open("questions", "wb") as fp:
    pickle.dump(questions, fp)
with open("remaining", "wb") as fp:
    pickle.dump(remaining, fp)