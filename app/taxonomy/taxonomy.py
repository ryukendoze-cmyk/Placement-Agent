import json
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

@dataclass(frozen=True)
class Skill:
    id: str
    name: str
    category: str
    aliases: tuple[str, ...] = field(default=())

class Taxonomy:
    def __init__(self, skills: list[dict] | list[Skill]):
        self._skills: list[Skill] = [
            s if isinstance(s, Skill) else Skill(s["id"], s["name"], s["category"], tuple(s.get("aliases", [])))
            for s in skills
        ]
        self._by_id = {s.id: s for s in self._skills}
        self.alias_map: dict[str, str] = {}
        for s in self._skills:
            self.alias_map[s.id.lower()] = s.id
            for a in s.aliases:
                self.alias_map[a.lower()] = s.id

    def resolve(self, token: str) -> str | None:
        return self.alias_map.get(token.lower())

    def get(self, skill_id: str) -> Skill:
        return self._by_id[skill_id]

    def get_opt(self, skill_id: str) -> Skill | None:
        return self._by_id.get(skill_id)

    def all(self) -> list[Skill]:
        return list(self._skills)

def load_taxonomy() -> Taxonomy:
    return Taxonomy(json.loads((DATA_DIR / "skills.json").read_text(encoding="utf-8")))
