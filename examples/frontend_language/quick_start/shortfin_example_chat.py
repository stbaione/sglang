"""
Usage:
# Prior to running this script, you need to have a Shortfin server running.
# Build:
#   https://github.com/nod-ai/SHARK-Platform/blob/main/shortfin/README.md
# Run:
#   https://github.com/nod-ai/SHARK-Platform/blob/main/shortfin/python/shortfin_apps/llm/README.md

python3 shortfin_example_chat.py --base_url <url_to_shortfin_server>
"""

import argparse
import os

import sglang as sgl


@sgl.function
def multi_turn_question(s, question_1, question_2):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1", max_tokens=256))
    s += sgl.user(question_2)
    s += sgl.assistant(sgl.gen("answer_2", max_tokens=256))


@sgl.function
def tip_suggestion(s):
    s += (
        "Here are two tips for staying healthy: "
        "1. Balanced Diet. 2. Regular Exercise.\n\n"
    )

    forks = s.fork(2)
    for i, f in enumerate(forks):
        f += f"Now, expand tip {i+1} into a paragraph:\n"
        f += sgl.gen(f"detailed_tip", max_tokens=256, stop="\n\n")

    s += "Tip 1:" + forks[0]["detailed_tip"] + "\n"
    s += "Tip 2:" + forks[1]["detailed_tip"] + "\n"
    s += "In summary" + sgl.gen("summary")


def forked_gen():
    state = tip_suggestion.run()
    print(state.text)
    # print("Tip 1:", state["detailed_tip"][0])
    # print("Tip 2:", state["detailed_tip"][1])

    # print("In summary:", state["summary"])


def single():
    state = multi_turn_question.run(
        question_1="What is the capital of the United States?",
        question_2="List two local attractions.",
    )

    for m in state.messages():
        print(m["role"], ":", m["content"])

    print("\n-- answer_1 --\n", state["answer_1"])
    print("\n-- answer_2 --\n", state["answer_2"])


def stream():
    state = multi_turn_question.run(
        question_1="What is the capital of the United States?",
        question_2="List two local attractions.",
        stream=True,
    )

    for out in state.text_iter():
        print(out, end="", flush=True)
    print()


def batch():
    states = multi_turn_question.run_batch(
        [
            {
                "question_1": "What is the capital of the United States?",
                "question_2": "List two local attractions.",
            },
            {
                "question_1": "What is the capital of France?",
                "question_2": "What is the population of this city?",
            },
        ]
    )

    for s in states:
        print(s.messages())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", default="http://localhost:8000")
    args = parser.parse_args()
    base_url = args.base_url

    backend = sgl.Shortfin(base_url=base_url)
    sgl.set_default_backend(backend)

    # Run Forked request
    print("\n========== forked_gen ==========\n")
    forked_gen()

    # Run a single request
    print("\n========== single ==========\n")
    single()

    # Stream output
    print("\n========== stream ==========\n")
    stream()

    # Batch request
    print("\n========== batch ==========\n")
    batch()
