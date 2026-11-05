from app.taxonomy.taxonomy import Taxonomy

DATA = [
    {"id": "python", "name": "Python", "category": "languages", "aliases": ["python"]},
    {"id": "js", "name": "JavaScript", "category": "web_frontend", "aliases": ["javascript"]},
]

def test_resolve_exact_and_alias():
    t = Taxonomy(DATA)
    assert t.resolve("Python") == "python"
    assert t.resolve("JAVASCRIPT") == "js"
    assert t.resolve("rust") is None

def test_alias_map_is_flattened():
    t = Taxonomy(DATA)
    assert t.alias_map["javascript"] == "js"
    assert t.alias_map["python"] == "python"
