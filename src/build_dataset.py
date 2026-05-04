import json
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from program_to_cot import program_to_chain_of_thought, get_program_depth


def infer_question_type(question):
    q = question.lower()
    if "how many" in q:
        return "count"
    if "there a" in q or "there an" in q or "any" in q:
        return "exist"
    if "more" in q or "fewer" in q or "less" in q:
        return "compare_int"
    if "same" in q or "equal" in q or "different" in q:
        return "compare_attr"
    return "query_attr"


def build_clevr_dataset(questions_path, output_path, max_samples=None):
    with open(questions_path) as f:
        questions = json.load(f)["questions"]

    if max_samples:
        questions = questions[:max_samples]

    records = []
    for q in questions:
        cot = program_to_chain_of_thought(q["program"])
        depth = get_program_depth(q["program"])
        qtype = infer_question_type(q["question"])

        records.append({
            "image_filename": q["image_filename"],
            "question": q["question"],
            "answer": str(q["answer"]).lower(),
            "chain_of_thought": cot,
            "cot_answer": f"{cot} Therefore, the answer is: {q['answer']}.",
            "question_type": qtype,
            "program_depth": depth,
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f)
    print(f"Saved {len(records)} records to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_questions", default="data/clevr/CLEVR_v1.0/questions/CLEVR_train_questions.json")
    parser.add_argument("--val_questions",   default="data/clevr/CLEVR_v1.0/questions/CLEVR_val_questions.json")
    parser.add_argument("--train_out",       default="data/processed/train.json")
    parser.add_argument("--val_out",         default="data/processed/val.json")
    parser.add_argument("--train_samples",   type=int, default=50000)
    parser.add_argument("--val_samples",     type=int, default=10000)
    args = parser.parse_args()

    print("Building training set...")
    build_clevr_dataset(args.train_questions, args.train_out, args.train_samples)

    print("Building validation set...")
    build_clevr_dataset(args.val_questions, args.val_out, args.val_samples)

    print("Done.")
