# ESCO Skill Extractor

This is a a tool that extract **ESCO skills from texts** such as job descriptions or CVs. It uses a special embedding model that allows prompts, called `instructor`.

## Installation

```bash
pip install esco-skill-extractor
```

## Usage

```python
from esco_skill_extractor import SkillExtractor

# `device` kwarg is optional and defaults to 'cpu', `cuda` or others can be used.
# `threshold` kwarg is optional and defaults to 0.8, it's the cosine similarity threshold.
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
#         {
#             "id": "bf4d884f-c848-402a-b130-69c266b04164",
#             "label": "apply basic programming skills"
#         }
#     ],
#     [
#         {
#             "id": "f0de4973-0a70-4644-8fd4-3a97080476f4",
#             "label": "DevOps"
#         },
#         {
#             "id": "1b2ec9bb-ba7c-4f93-87ac-ec712c9b68c3", 
#             "label": "install containers"
#         },
#         {
#             "id": "6b643893-0a1f-4f6c-83a1-e7eef75849b9",
#             "label": "develop with cloud services"
#         }
#     ]
# ]
```

## Considerations

While there's been some effort to make the model ignore irrelevant information such as company names, contact information, recruitment hustle and others, the model still tries to extract skills from them sometimes.

For instance a salary range could be interpreted as a skill `determine salaries`.

It is advised to clean the texts before passing them to the model if possible.

## How it works

1. It creates embeddings from esco skills found in the official ESCO website.
2. It creates embeddings from the input text (one for each sentence).
3. It compares the embeddings of the text with the embeddings of the ESCO skills using cosine similarity.
4. It returns the most similar esco skill per sentence if its similarity passes a predefined threshold.

## References

-   [Instructor model](https://huggingface.co/hkunlp/instructor-base)
-   [ESCO](https://ec.europa.eu/esco/portal/home)
