# ESCO Skill Extractor

This is a a tool that extract **ESCO skills from texts** such as job descriptions or CVs. It uses a transformer and compares its embedding using cosine similarity. 

## Installation

```bash
pip install esco-skill-extractor
```

## Usage

```python
from esco_skill_extractor import SkillExtractor

# `device` kwarg is optional and defaults to 'cpu', `cuda` or others can be used.
# `threshold` kwarg is optional and defaults to 0.4, it's the cosine similarity threshold.
skill_extractor = SkillExtractor()

ads = [
    "We are looking for a software engineer with experience in Java and Python.",
    "We are looking for a devops engineer. Containerization tools such as Docker is a must. AWS is a plus."
    # ...
]

print(skill_extractor.get_skills(ads))

# Output:
# [
#     [
#         "http://data.europa.eu/esco/skill/ccd0a1d9-afda-43d9-b901-96344886e14d"
#     ],
#     [
#         "http://data.europa.eu/esco/skill/f0de4973-0a70-4644-8fd4-3a97080476f4",
#         "http://data.europa.eu/esco/skill/ae4f0cc6-e0b9-47f5-bdca-2fc2e6316dce",
#     ],
# ]
# ]
```

## How it works

1. It creates embeddings from esco skills found in the official ESCO website.
2. It creates embeddings from the input text (one for each sentence).
3. It compares the embeddings of the text with the embeddings of the ESCO skills using cosine similarity.
4. It returns the most similar esco skill per sentence if its similarity passes a predefined threshold.
