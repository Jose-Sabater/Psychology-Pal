import os
from openai import OpenAI
from typing import List, Dict
import logging
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed

load_dotenv()


class OpenAIAssistant:
    def __init__(self, model: str = "gpt-3.5-turbo", **kwargs):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    @retry(stop=stop_after_attempt(4), wait=wait_fixed(2))
    def get_openai_completion(
        self,
        system_message: str,
        user_messages: List[str],
        assistant_messages: List[str],
    ) -> Dict:
        """Invoke OpenAI's ChatCompletion with the provided messages."""
        messages = [
            {"role": "system", "content": f"{system_message}"},
            *self.user_assistant_message(user_messages, assistant_messages),
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
            )
            return response

        except Exception as e:
            logging.error(f"Error invoking OpenAI's ChatCompletion : {e}")
            raise

    @staticmethod
    def user_assistant_message(
        user_messages: List[str], assistant_messages: List[str]
    ) -> List[Dict[str, str]]:
        """Returns a list of dictionaries of the user and assistant messages."""
        user_assistant_messages = []
        assistant_count = len(assistant_messages)

        # Iterate over all user messages
        for i, user_message in enumerate(user_messages):
            user_assistant_messages.append(
                {"role": "user", "content": f"{user_message}"}
            )

            # Only add an assistant message if one exists for the current user message
            if i < assistant_count:
                assistant_message = assistant_messages[i]
                user_assistant_messages.append(
                    {"role": "assistant", "content": f"{assistant_message}"}
                )

        return user_assistant_messages
