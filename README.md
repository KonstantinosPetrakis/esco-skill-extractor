# ESCO Skill Extractor

This is a a tool that extract **ESCO skills from texts** such as job descriptions or CVs. It uses a transformer and compares its embedding using cosine similarity.

## Installation

```bash
pip install esco-skill-extractor
```

or for Nvidia GPU acceleration:

```bash
pip install esco-skill-extractor[cuda]
```

## Usage

### Via python

```python
from esco_skill_extractor import SkillExtractor

# Don't be scared, the 1st time will take longer to download the model and create the embeddings.
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
```

### Via GUI

```bash
# Visit the URL printed in the console.
# run python -m esco_skill_extractor --help for more options.
python -m esco_skill_extractor 
```

<img src="docs/gui.gif">

### Via API

```bash
# Visit the URL printed in the console.
# run python -m esco_skill_extractor --help for more options.
python -m esco_skill_extractor 
```

```js
async function getSkills() {
    const texts = [
        "We are looking for a software engineer with experience in Java and Python.",
        "We are looking for a devops engineer. Containerization tools such as Docker is a must. AWS is a plus."
        // ...
    ];

    // Default host is localhost, and default port is 8000. Check CLI options for more.
    const response = await fetch("http://localhost:8000/extract", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(texts),
    });

    const skills = await response.json();
    console.log(skills);
    // Output:
    // [
    //     [
    //         "http://data.europa.eu/esco/skill/ccd0a1d9-afda-43d9-b901-96344886e14d"
    //     ],
    //     [
    //         "http://data.europa.eu/esco/skill/f0de4973-0a70-4644-8fd4-3a97080476f4",
    //         "http://data.europa.eu/esco/skill/ae4f0cc6-e0b9-47f5-bdca-2fc2e6316dce",
    //     ],
    // ]
}
```

## Possible keyword arguments for `SkillExtractor`

| Keyword Argument | Description                                                                                                            | Default |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------- | ------- |
| threshold        | Skills surpassing this cosine similarity threshold are considered a match.                                             | 0.4     |
| device           | The device where the copulations will take place. E.g torch device.                                                    | "cpu"   |
| max_words        | If any sentence in the input surpasses the set word_length considerably, its summarized close to that number of words. | -1      |

## How it works

1. It creates embeddings from esco skills found in the official ESCO website.
2. It creates embeddings from the input text (one for each sentence).
   1. If any sentence surpasses the `max_words` limit, it is summarized to that number of words by using an [implementation of the TextRank algorithm](https://github.com/summanlp/textrank).
3. It compares the embeddings of the text with the embeddings of the ESCO skills using cosine similarity.
4. It returns the most similar esco skill per sentence if its similarity passes a predefined threshold.
