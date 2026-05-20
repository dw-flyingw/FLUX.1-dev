"""Prompt extension using LLM."""

import os
from typing import Optional

from openai import OpenAI


class PromptExtendError(Exception):
    """Error during prompt extension."""

    pass


class PromptExtender:
    """Extend prompts using an LLM."""

    def __init__(self, model_url: str):
        """Initialize with OpenAI-compatible endpoint.
        
        Args:
            model_url: OpenAI-compatible API endpoint
        """
        self.model_url = model_url
        self.client = OpenAI(
            base_url=model_url,
            api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
        )

    def extend(self, prompt: str) -> str:
        """Extend a prompt with more detail.
        
        Args:
            prompt: Original prompt
            
        Returns:
            Extended prompt
            
        Raises:
            PromptExtendError: If extension fails
        """
        system_prompt = (
            "You are an expert at creating detailed image generation prompts. "
            "Expand the user's brief prompt into a rich, detailed description "
            "that will produce high-quality images. Include details about: "
            "lighting, composition, style, mood, colors, and specific visual elements. "
            "Keep it under 500 characters."
        )
        
        try:
            response = self.client.chat.completions.create(
                model="default",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Expand this prompt: {prompt}"},
                ],
                max_tokens=200,
                temperature=0.7,
            )
            extended = response.choices[0].message.content.strip()
            return extended or prompt
        except Exception as e:
            raise PromptExtendError(f"Failed to extend prompt: {e}") from e
