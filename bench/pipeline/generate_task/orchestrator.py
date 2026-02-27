from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
from bench.pipeline.generate_task.task_branch_base import TaskBranch

class TaskOrchestrator:
    def __init__(
        self,
        store: Dict[str, Any],
        branches: List[TaskBranch],
        save_path: Optional[str | Path] = None,
        runs_per_branch: Union[int, Dict[str, int]] = 5,
    ):
        self.store = store
        self.branches = branches
        self.tasks: List[Dict[str, Any]] = []
        self.save_path = Path(save_path) if save_path else None
        # allow global int or per-branch override
        self.runs_per_branch: Dict[str, int] = (
            {b.branch_name: runs_per_branch for b in branches}
            if isinstance(runs_per_branch, int)
            else runs_per_branch
        )
        self.tasks_by_branch: Dict[str, List[Any]] = {}

    def run(self) -> Dict[str, Any]:
        """
        Run all branches sequentially.
        """
        skipped = []
        for branch in self.branches:
            if hasattr(branch, "is_applicable") and not branch.is_applicable(self.store):
                skipped.append(branch.branch_name)
                continue
            runs = self.runs_per_branch.get(branch.branch_name, 5)
            branch_tasks: List[Any] = []
            for _ in range(runs):
                new_tasks = branch.run(self.store)
                branch_tasks.extend(new_tasks)
            self.tasks_by_branch[branch.branch_name] = branch_tasks
            self.tasks.extend(branch_tasks)

        result = {
            "final_store": self.store,
            "task_instructions": self.tasks,
            "tasks_by_branch": self.tasks_by_branch,
            "skipped_branches": skipped,
        }

        if self.save_path:
            self.save(self.save_path)

        return result

    def build_combine_prompt(self, tasks_by_branch: Dict[str, List[Any]]) -> str:
        """
        Build a prompt for downstream LLM combination of sampled tasks across branches.
        This does not execute combination; it only prepares instructions.
        """
        return (
            "You are asked to synthesize a concise set of user-facing task instructions "
            "by selecting and combining tasks generated from multiple branches.\n\n"
            "Inputs (JSON):\n"
            f"{json.dumps(tasks_by_branch, ensure_ascii=False, indent=2)}\n\n"
            "Requirements:\n"
            "- Preserve task diversity across branches; avoid duplicates.\n"
            "- Keep each combined task actionable and self-contained.\n"
            "- Do not mention internal branch names, tools, or system internals.\n"
            "- If two tasks are near-duplicate, keep the clearer one.\n"
            "- Output a JSON array of tasks, each with: task_instruction, label, intent.\n"
        )

    def save(self, path: Path) -> None:
        """
        Persist generated tasks (and lightweight store context) to JSON.
        DataFrames from wearable tables are intentionally excluded.
        """
        payload = {
            "store_meta": self.store.get("meta", {}),
            "profile": self.store.get("profile", {}),
            "tasks": self.tasks,
            "tasks_by_branch": self.tasks_by_branch,
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
