from typing import List
import warnings
import pickle
import os

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch


class SkillExtractor:
    def __init__(
        self,
        skills_threshold: float = 0.4,
        occupation_threshold=0.45,
        device: str = "cpu",
    ):
        """
        Loads the model, skills and skill embeddings.

        Args:
            threshold (float, optional): The similarity threshold for skill comparisons. Increase it to be more harsh. Defaults to 0.4. Range: [0, 1].
            device (str, optional): The device where the model will run. Defaults to "cpu".
            max_words (int, optional): If the inputted text is longer than this number of words, it will be summarized close to this number of words. Defaults to -1 (no summarization).
        """

        self.skills_threshold = skills_threshold
        self.occupation_threshold = occupation_threshold
        self.device = device
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

        Returns:
            List[List[str]]: A list of lists containing the IDs of the entities for each text.
        """

        # Calculate the embeddings for all texts
        text_embeddings = self._model.encode(
            texts,
            device=self.device,
            normalize_embeddings=True,
            convert_to_tensor=True,
        )

        # Calculate the similarity between all texts and all entities and find entities surpassing the threshold for each text
        similarity_matrix = util.dot_score(text_embeddings, entity_embeddings)

        entity_ids_per_text = []
        for similarity_scores in similarity_matrix:
            entity_indices = (
                torch.nonzero(similarity_scores > threshold)
                .squeeze(dim=-1)
                .unique()
                .tolist()
            )
            entity_ids_per_text.append(np.take(entity_ids, entity_indices).tolist())

        return entity_ids_per_text

    def get_skills(self, texts: List[str]) -> List[List[str]]:
        """
        This method extracts the ESCO skills from the texts.

        Returns:
            List[List[str]]: A list of lists containing the IDs of the skills for each text.
        """

        return self._get_entity(
            texts, self._skill_ids, self._skill_embeddings, self.skills_threshold
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
