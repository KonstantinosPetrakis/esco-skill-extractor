from itertools import chain
from typing import Union, List
import warnings
import pickle
import os

from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import spacy
import torch


class SkillExtractor:
    def __init__(
        self,
        skills_threshold: float = 0.6,
        occupation_threshold: float = 0.55,
        device: Union[str, None] = None,
    ):
        """
        Loads the models, skills and skill embeddings.

        Args:
            skills_threshold (float, optional): The similarity threshold for skill comparisons. Increase it to be more harsh. Defaults to 0.45. Range: [0, 1].
            occupation_threshold (float, optional): The similarity threshold for occupation comparisons. Increase it to be more harsh. Defaults to 0.55. Range: [0, 1].
            device (Union[str, None], optional): The device where the model will run. Defaults to "cuda" if available, otherwise "cpu".
        """

        self.skills_threshold = skills_threshold
        self.occupation_threshold = occupation_threshold
        self.device = (
            device if device else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._dir = __file__.replace("__init__.py", "")
        self._load_models()
        self._load_skills()
        self._load_occupations()
        self._create_skill_embeddings()
        self._create_occupation_embeddings()

    def _load_models(self):
        """
        This method loads the model from the SentenceTransformer library.
        """

        # Ignore the security warning messages about loading the model from pickle
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

        self._skillner = spacy.load("en_skillner")

    def _load_skills(self):
        """
        This method loads the skills from the skills.csv file.
        """

        self._skills = pd.read_csv(f"{self._dir}/data/skills.csv")
        self._skill_ids = self._skills["id"].to_numpy()

    def _load_occupations(self):
        """
        This method loads the occupations from the occupations.csv file.
        """

        self._occupations = pd.read_csv(f"{self._dir}/data/occupations.csv")
        self._occupation_ids = self._occupations["id"].to_numpy()

    def _create_skill_embeddings(self):
        """
        This method creates the skill embeddings and saves them to a cache file.
        If the cache file exists, it loads the embeddings from it.
        """

        if os.path.exists(f"{self._dir}/data/skill_embeddings.bin"):
            with open(f"{self._dir}/data/skill_embeddings.bin", "rb") as f:
                self._skill_embeddings = pickle.load(f).to(self.device)
        else:
            print(
                "Skill embeddings file not found. Creating embeddings from scratch..."
            )
            self._skill_embeddings = self._model.encode(
                self._skills["description"].to_list(),
                device=self.device,
                normalize_embeddings=True,
                convert_to_tensor=True,
            )
            with open(f"{self._dir}/data/skill_embeddings.bin", "wb") as f:
                pickle.dump(self._skill_embeddings, f)

    def _create_occupation_embeddings(self):
        """
        This method creates the occupations embeddings and saves them to a cache file.
        If the cache file exists, it loads the embeddings from it.
        """

        if os.path.exists(f"{self._dir}/data/occupation_embeddings.bin"):
            with open(f"{self._dir}/data/occupation_embeddings.bin", "rb") as f:
                self._occupation_embeddings = pickle.load(f).to(self.device)
        else:
            print(
                "Occupation embeddings file not found. Creating embeddings from scratch..."
            )
            self._occupation_embeddings = self._model.encode(
                self._occupations["description"].to_list(),
                device=self.device,
                normalize_embeddings=True,
                convert_to_tensor=True,
            )
            with open(f"{self._dir}/data/occupation_embeddings.bin", "wb") as f:
                pickle.dump(self._occupation_embeddings, f)

    def _texts_to_tokens(self, texts: List[str]) -> List[List[str]]:
        """
        This method splits the texts into tokens.

        Args:
            text (str): The texts to be split.

        Returns:
            List[str]: A list of lists containing the tokens for each text.
        """

        docs = self._skillner.pipe(texts)
        return [[ent.text.strip() for ent in doc.ents] for doc in docs]

    def _get_entity(
        self,
        texts: List[str],
        entity_ids: np.ndarray[str],
        entity_embeddings: torch.Tensor,
        threshold: float,
    ) -> List[List[str]]:
        """
        This method extracts the entities from the texts.

        Args:
            texts (List[str]): The texts from which the entities will be extracted.
            entity_ids (np.ndarray[str]): The IDs of the entities.
            entity_embeddings (torch.Tensor): The embeddings of the entities.
            threshold (float): The similarity threshold for entity comparisons. Increase it to be more harsh.

        Returns:
            List[List[str]]: A list of lists containing the IDs of the entities for each text.
        """

        # If there are no texts, return an empty list
        if all(not text for text in texts):
            return [[] for _ in texts]

        # Split the texts into tokens and then flatten them to perform calculations faster
        texts = self._texts_to_tokens(texts)
        tokens = list(chain.from_iterable(texts))

        # If there are no tokens, return an empty list
        if not tokens:
            return [[] for _ in texts]

        # Calculate the embeddings for all flattened tokens
        sentence_embeddings = self._model.encode(
            tokens,
            device=self.device,
            normalize_embeddings=True,
            convert_to_tensor=True,
        )

        # Calculate the similarity between all flattened tokens and all entities and
        # find the most similar entity for each sentence.
        # The embeddings are normalized so the dot product is the cosine similarity
        similarity_matrix = util.dot_score(sentence_embeddings, entity_embeddings)
        most_similar_entity_scores, most_similar_entity_indices = torch.max(
            similarity_matrix, dim=-1
        )

        # Un-flatten the list of most similar entities to match the original texts
        entity_ids_per_text = []
        sentences = 0

        for text in texts:
            sentences_in_text = len(text)

            most_similar_entity_indices_text = most_similar_entity_indices[
                sentences : sentences + sentences_in_text
            ]
            most_similar_entity_scores_text = most_similar_entity_scores[
                sentences : sentences + sentences_in_text
            ]

            # Filter the entities based on the threshold
            most_similar_entity_indices_text = (
                most_similar_entity_indices_text[
                    torch.nonzero(most_similar_entity_scores_text > threshold)
                ]
                .squeeze(dim=-1)
                .unique()
                .tolist()
            )

            # Create a list of dictionaries containing the entities for the current text
            entity_ids_per_text.append(
                np.take(entity_ids, most_similar_entity_indices_text).tolist()
            )

            sentences += sentences_in_text

        return entity_ids_per_text

    def get_skills(self, texts: List[str]) -> List[List[str]]:
        """
        This method extracts the ESCO skills from the texts.

        Returns:
            List[List[str]]: A list of lists containing the IDs of the skills for each text.
        """

        return self._get_entity(
            texts,
            self._skill_ids,
            self._skill_embeddings,
            self.skills_threshold,
        )

    def get_occupations(self, texts: List[str]) -> List[List[str]]:
        """
        This method extracts the ESCO occupations from the texts.

        Returns:
            List[List[str]]: A list of lists containing the IDs of the occupations for each text.
        """

        return self._get_entity(
            texts,
            self._occupation_ids,
            self._occupation_embeddings,
            self.occupation_threshold,
        )
