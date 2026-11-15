"""Rebuild supplemental offline questions from existing factual skill references.

Preserves original questions and IDs. Adds a mechanism, troubleshooting and
design scenario per canonical skill, with the original reference mechanisms as
grading anchors. This is an offline authoring utility, never a live AI path.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def expand():
    path = ROOT / "data/questions.json"
    questions = json.loads(path.read_text(encoding="utf-8"))
    questions = [q for q in questions if not q["id"].startswith("q-offline-")]
    skills = json.loads((ROOT / "data/skills.json").read_text(encoding="utf-8"))
    roles = {"web_frontend": "frontend", "data_ml": "data", "devops_cloud": "devops"}
    for skill in skills:
        refs = [q for q in questions if q["skill_id"] == skill["id"]]
        if not refs:
            raise ValueError(f"Add a factual reference for {skill['id']} before expanding")
        refs.sort(key=lambda q: (q["difficulty"], q["id"]))
        for difficulty in range(3):
            ref = min(refs, key=lambda q: abs(q["difficulty"] - difficulty))
            concepts = list(dict.fromkeys(ref["keywords"]))
            first, second = concepts[:2]
            name = skill["name"]
            if skill["category"] == "soft_skills":
                scenarios = [
                    f"Give a specific example of {name} from a team task. How did you put {first} and {second} into practice, and what observable result followed?",
                    f"A team task breaks down because of weak {name}. Using {first} and {second}, walk through your intervention and how you would check whether it worked.",
                    f"Two stakeholders disagree about how to apply {name} under a tight deadline. How would you balance {first} against {second}, justify your decision, and measure the outcome?",
                ]
            else:
                scenarios = [
                    f"For a small task using {name}, demonstrate how {first} relates to {second}. Walk through a concrete example, its expected behavior, and a check that would expose an incorrect implementation.",
                    f"An implementation using {name} behaves incorrectly around {first} and {second}. What evidence would you collect, how would you isolate the cause, and what regression check would confirm your fix?",
                    f"You must choose an approach using {name} under stricter reliability and resource constraints. Compare the implications of {first} and {second}, explain the mechanisms involved, and justify a design with measurable validation.",
                ]
            outline = list(ref["outline"]) + [
                "Apply the reference mechanisms to a concrete example with expected behavior",
                "Use a reproducible check or observable outcome to support the decision",
            ]
            questions.append({"id": f"q-offline-{skill['id']}-{difficulty}",
                              "role": roles.get(skill["category"], ref["role"]),
                              "skill_id": skill["id"], "difficulty": difficulty,
                              "question": scenarios[difficulty], "keywords": concepts, "outline": outline})
    path.write_text(json.dumps(questions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Offline bank: {len(questions)} questions; {len(skills)} canonical skills with three new scenarios each")


if __name__ == "__main__":
    expand()
