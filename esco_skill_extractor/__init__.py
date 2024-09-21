from typing import List, Dict, Union, Literal
import warnings
import pickle
import os
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np


class SkillExtractor:
    def __init__(self, threshold: float = 0.5, device: str = "cpu"):
        """
        Loads the model, skills and skill embeddings.

        Args:
            threshold (float, optional): The similarity threshold for skill comparisons. Increase it to more harsh. Defaults to 0.85. Range: [0, 1].
            device (str, optional): The device where the model will run. Defaults to "cpu".
        """

        self.threshold = threshold
        self.device = device
        self._dir = __file__.replace("__init__.py", "")
        self._load_models()
        self._load_skills()
        self._create_skill_embeddings()

    def _load_models(self):
        """
        This method loads the model from the SentenceTransformer library.
        """

        # Ignore the security warning messages about loading the model from pickle
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

    def _load_skills(self):
        """
        This method loads the skills from the skills.csv file.
        """

        self._skills = pd.read_csv(f"{self._dir}/data/skills.csv")

    def _create_skill_embeddings(self):
        """
        This method creates the skill embeddings and saves them to a cache file.
        If the cache file exists, it loads the embeddings from it.
        """

        if os.path.exists(f"{self._dir}/data/embeddings.bin"):
            with open(f"{self._dir}/data/embeddings.bin", "rb") as f:
                self._skill_embeddings = pickle.load(f)
        else:
            print(
                "Skill embeddings file not found. Creating embeddings from scratch..."
            )
            self._skill_embeddings = self._model.encode(self._skills["label"].to_list())
            with open(f"{self._dir}/data/embeddings.bin", "wb") as f:
                pickle.dump(self._skill_embeddings, f)

    def _text_to_sentences(self, text: str) -> List[str]:
        """
        This method splits the text into sentences.

        Args:
            text (str): The text to split into sentences.

        Returns:
            List[str]: A list of sentences.
        """

        # Ignore short sentences such as '.e.g', '.etc' and stuff
        return list(
            filter(
                lambda x: len(x) > 5,
                re.split(r"(\r|\n|\t|\.|;)+", text),
            )
        )

    def get_skills(
        self, texts: List[str]
    ) -> List[List[Dict[Union[Literal["id"], Literal["label"]], str]]]:
        """
        This method extracts the ESCO skills from the texts.

        Returns:
            List[List[Dict[Union[Literal["id"], Literal["label"]], str]]]: A list of lists containing the skills for each text.
        """

        # Flatten the list of sentences so similarity can be calculated in one operation for all sentences
        texts = [self._text_to_sentences(text) for text in texts]
        all_sentences = [sentence for text in texts for sentence in text]

        # Calculate the embeddings for all sentences
        all_sentences_embeddings = np.array(self._model.encode(all_sentences))

        # Calculate the similarity between all sentences and all skills and find the most similar skill for each sentence and its similarity score
        similarity_matrix = cosine_similarity(
            all_sentences_embeddings, self._skill_embeddings
        )
        most_similar_skills_indices = np.argmax(similarity_matrix, axis=1)
        most_similar_skills_scores = np.max(similarity_matrix, axis=1)

        # Unflatten the list of most similar skills indices to match the original texts
        skills_indices_per_text = []
        sentences = 0
        for text in texts:
            text_sentences = len(text)

            # Get the most similar skills and their scores for the current text
            most_similar_skills_indices_text = most_similar_skills_indices[
                sentences : sentences + text_sentences
            ]
            most_similar_skills_scores_text = most_similar_skills_scores[
                sentences : sentences + text_sentences
            ]

            # Filter the skills based on the threshold
            most_similar_skills_indices_text = most_similar_skills_indices_text[
                np.where(most_similar_skills_scores_text > self.threshold)
            ]

            # Create a list of dictionaries containing the skills for the current text
            skills_indices_per_text.append(
                self._skills.take(most_similar_skills_indices_text).to_dict(
                    orient="records"
                )
            )

            sentences += text_sentences

        return skills_indices_per_text
