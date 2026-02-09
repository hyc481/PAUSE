from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

from bench.backend.utils.clients import call_llm

class TaskBranch(ABC):
    branch_name: str

    def __init__(self, client, model):
        self.client = client
        self.model = model


    @abstractmethod
    def build_prompt(self, store: Dict[str, Any]) -> str:
        """
        Compose branch-specific generation prompt
        (base prompt + branch instructions).
        """
        pass

    def is_applicable(self, store: Dict[str, Any]) -> bool:
        """
        Optional precondition check to decide whether this branch should run
        under the given store context. Default: always applicable.
        Override in subclasses when branch requires certain data availability.
        """
        return True

    def call_llm(self, prompt: str) -> str:
        """
        All branches share the same LLM calling convention.
        """
        messages = [{"role": "system", "content": prompt}]
        return call_llm(
            self.client,
            model=self.model,
            messages=messages,
            temperature=self._llm_temperature(self.model),
        )

    @staticmethod
    def _llm_temperature(model: str) -> Optional[float]:
        if model.startswith("gpt-5"):
            return None
        return 0

    @abstractmethod
    def postprocess(
        self,
        llm_output: str,
        store: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Parse LLM output → task_instruction(s).
        May mutate store IN-PLACE.
        """
        pass

    def run(self, store: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = self.build_prompt(store)
        llm_output = self.call_llm(prompt)
        return self.postprocess(llm_output, store)
