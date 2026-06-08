# core/skill_registry.py
import os
import yaml
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Skill:
    name: str
    description: str
    content: str  # The markdown body containing the actual rules


class SkillRegistry:
    def __init__(self, skills_dir: str):
        self.skills_dir = skills_dir
        self.skills: Dict[str, Skill] = {}
        self._load_skills()

    def _load_skills(self):
        """Scan the skills directory and parse all SKILL.md files."""
        if not os.path.exists(self.skills_dir):
            print(f"Warning: Skills directory '{self.skills_dir}' not found.")
            return

        for root, _, files in os.walk(self.skills_dir):
            for file in files:
                if file.endswith(".md"):
                    filepath = os.path.join(root, file)
                    self._parse_skill_file(filepath)

    def _parse_skill_file(self, filepath: str):
        """Extract YAML frontmatter (meta) and Markdown body (content)."""
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        # Split by '---' to separate frontmatter from body
        parts = raw_text.split('---', 2)
        if len(parts) >= 3:
            try:
                meta = yaml.safe_load(parts[1])
                name = meta.get('name', 'unknown-skill')
                description = meta.get('description', '')
                content = parts[2].strip()

                self.skills[name] = Skill(name=name, description=description, content=content)
                print(f"Successfully loaded skill: {name}")
            except yaml.YAMLError as e:
                print(f"Error parsing YAML in {filepath}: {e}")

    def get_metadata_prompt(self) -> str:
        """Format the skill list for the Router LLM to read."""
        if not self.skills:
            return "No skills available."
        return "\n".join([f"- {name}: {skill.description}" for name, skill in self.skills.items()])

    def get_skill(self, name: str) -> Optional[Skill]:
        """Retrieve the full skill object by its name."""
        return self.skills.get(name)