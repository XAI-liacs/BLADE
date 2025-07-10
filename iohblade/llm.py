"""
LLM modules to connect to different LLM providers. Also extracts code, name and description.
"""

import re
import time
from abc import ABC, abstractmethod
import logging
import copy

import google.generativeai as genai
import ollama
import openai
from ConfigSpace import ConfigurationSpace
from tokencost import (
    calculate_completion_cost,
    calculate_prompt_cost,
    count_message_tokens,
    count_string_tokens,
)

from .solution import Solution
from .utils import NoCodeException


class LLM(ABC):
    def __init__(
        self,
        api_key,
        model="",
        base_url="",
        code_pattern=None,
        name_pattern=None,
        desc_pattern=None,
        cs_pattern=None,
        logger=None,
    ):
        """
        Initializes the LLM manager with an API key, model name and base_url.

        Args:
            api_key (str): api key for authentication.
            model (str, optional): model abbreviation.
            base_url (str, optional): The url to call the API from.
            code_pattern (str, optional): The regex pattern to extract code from the response.
            name_pattern (str, optional): The regex pattern to extract the class name from the response.
            desc_pattern (str, optional): The regex pattern to extract the description from the response.
            cs_pattern (str, optional): The regex pattern to extract the configuration space from the response.
            logger (Logger, optional): A logger object to log the conversation.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.logger = logger
        self.log = self.logger != None
        self.code_pattern = (
            code_pattern if code_pattern != None else r"```(?:python)?\n(.*?)\n```"
        )
        self.name_pattern = (
            name_pattern
            if name_pattern != None
            else "class\\s*(\\w*)(?:\\(\\w*\\))?\\:"
        )
        self.desc_pattern = (
            desc_pattern if desc_pattern != None else r"#\s*Description\s*:\s*(.*)"
        )
        self.cs_pattern = (
            cs_pattern
            if cs_pattern != None
            else r"space\s*:\s*\n*```\n*(?:python)?\n(.*?)\n```"
        )
        # Mute tokecost logging
        logging.getLogger("tokencost").setLevel(logging.ERROR)

    @abstractmethod
    def _query(self, session: list):
        """
        Sends a conversation history to the configured model and returns the response text.

        Args:
            session (list of dict): A list of message dictionaries with keys
                "role" (e.g. "user", "assistant") and "content" (the message text).

        Returns:
            str: The text content of the LLM's response.
        """
        pass

    def query(self, session: list):
        """
        Sends a conversation history to the configured model and returns the response text.

        Args:
            session_messages (list of dict): A list of message dictionaries with keys
                "role" (e.g. "user", "assistant") and "content" (the message text).

        Returns:
            str: The text content of the LLM's response.
        """
        if self.log:
            try:
                cost = calculate_prompt_cost(session, self.model)
            except Exception:
                cost = 0
            try:
                tokens = count_message_tokens(session, model=self.model)
            except Exception:
                tokens = 0
            self.logger.log_conversation(
                "client",
                "\n".join([d["content"] for d in session]),
                cost,
                tokens,
            )

        message = self._query(session)

        if self.log:
            try:
                cost = calculate_completion_cost(message, self.model)
            except Exception:
                cost = 0
            try:
                tokens = count_string_tokens(prompt=message, model=self.model)
            except Exception:
                tokens = 0
            self.logger.log_conversation(self.model, message, cost, tokens)

        return message

    def set_logger(self, logger):
        """
        Sets the logger object to log the conversation.

        Args:
            logger (Logger): A logger object to log the conversation.
        """
        self.logger = logger
        self.log = True

    def sample_solution(self, session_messages: list, parent_ids=[], HPO=False):
        """
        Interacts with a language model to generate or mutate solutions based on the provided session messages.

        Args:
            session_messages (list): A list of dictionaries with keys 'role' and 'content' to simulate a conversation with the language model.
            parent_ids (list, optional): The id of the parent the next sample will be generated from (if any).
            HPO (boolean, optional): If HPO is enabled, a configuration space will also be extracted (if possible).

        Returns:
            tuple: A tuple containing the new algorithm code, its class name, its full descriptive name and an optional configuration space object.

        Raises:
            NoCodeException: If the language model fails to return any code.
            Exception: Captures and logs any other exceptions that occur during the interaction.
        """
        message = self.query(session_messages)

        code = self.extract_algorithm_code(message)
        name = self.extract_classname(code)
        desc = self.extract_algorithm_description(message)
        cs = None
        if HPO:
            cs = self.extract_configspace(message)
        new_individual = Solution(
            name=name,
            description=desc,
            configspace=cs,
            code=code,
            parent_ids=parent_ids,
        )

        return new_individual

    def extract_classname(self, code):
        """Extract the Python class name from a given code string (if possible).

        Args:
            code (string): The code string to extract from.

        Returns:
            classname (string): The Python class name or empty string.
        """
        try:
            return re.findall(
                "class\\s*(\\w*)(?:\\(\\w*\\))?\\:",
                code,
                re.IGNORECASE,
            )[0]
        except Exception as e:
            return ""

    def extract_configspace(self, message):
        """
        Extracts the configuration space definition in json from a given message string using regular expressions.

        Args:
            message (str): The message string containing the algorithm code.

        Returns:
            ConfigSpace: Extracted configuration space object.
        """
        pattern = r"space\s*:\s*\n*```\n*(?:python)?\n(.*?)\n```"
        c = None
        for m in re.finditer(pattern, message, re.DOTALL | re.IGNORECASE):
            try:
                c = ConfigurationSpace(eval(m.group(1)))
            except Exception as e:
                pass
        return c

    def extract_algorithm_code(self, message):
        """
        Extracts algorithm code from a given message string using regular expressions.

        Args:
            message (str): The message string containing the algorithm code.

        Returns:
            str: Extracted algorithm code.

        Raises:
            NoCodeException: If no code block is found within the message.
        """
        pattern = r"```(?:python)?\n(.*?)\n```"
        match = re.search(pattern, message, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1)
        else:
            return """raise Exception("Could not extract generated code. The code should be encapsulated with ``` in your response.")"""  # trick to later raise this exception when the algorithm is evaluated.

    def extract_algorithm_description(self, message):
        """
        Extracts algorithm description from a given message string using regular expressions.

        Args:
            message (str): The message string containing the algorithm name and code.

        Returns:
            str: Extracted algorithm name or empty string.
        """
        pattern = r"#\s*Description\s*:\s*(.*)"
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return match.group(1)
        else:
            return ""

    def to_dict(self):
        """
        Returns a dictionary representation of the LLM including all parameters.

        Returns:
            dict: Dictionary representation of the LLM.
        """
        return {
            "model": self.model,
            "code_pattern": self.code_pattern,
            "name_pattern": self.name_pattern,
            "desc_pattern": self.desc_pattern,
            "cs_pattern": self.cs_pattern,
        }


class OpenAI_LLM(LLM):
    """
    A manager class for handling requests to OpenAI's GPT models.
    """

    def __init__(self, api_key, model="gpt-4-turbo", temperature=0.8, **kwargs):
        """
        Initializes the LLM manager with an API key and model name.

        Args:
            api_key (str): api key for authentication.
            model (str, optional): model abbreviation. Defaults to "gpt-4-turbo".
                Options are: gpt-3.5-turbo, gpt-4-turbo, gpt-4o, and others from OpeNAI models library.
        """
        super().__init__(api_key, model, None, **kwargs)
        self._client_kwargs = dict(api_key=api_key)
        self.client = openai.OpenAI(**self._client_kwargs)
        logging.getLogger("openai").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.ERROR)
        self.temperature = temperature

    def _query(self, session_messages, max_retries: int = 5, default_delay: int = 10):
        """
        Sends a conversation history to the configured model and returns the response text.

        Args:
            session_messages (list of dict): A list of message dictionaries with keys
                "role" (e.g. "user", "assistant") and "content" (the message text).

        Returns:
            str: The text content of the LLM's response.
        """

        attempt = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=session_messages,
                    temperature=self.temperature,
                )
                return response.choices[0].message.content

            except openai.RateLimitError as err:
                attempt += 1
                if attempt > max_retries:
                    raise

                retry_after = None
                if getattr(err, "response", None) is not None:
                    retry_after = err.response.headers.get("Retry-After")

                wait = int(retry_after) if retry_after else default_delay * attempt
                time.sleep(wait)

            except (
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.APIError,
            ) as err:
                attempt += 1
                if attempt > max_retries:
                    raise
                time.sleep(default_delay * attempt)

    # ---------- pickling / deepcopy helpers ----------
    def __getstate__(self):
        """Return the picklable part of the instance."""
        state = self.__dict__.copy()
        state.pop("client", None)  # the client itself is NOT picklable
        return state  # everything else is fine

    def __setstate__(self, state):
        """Restore from a pickled state."""
        self.__dict__.update(state)  # put back the simple stuff
        self.client = openai.OpenAI(
            **self._client_kwargs
        )  # rebuild non-picklable handle

    def __deepcopy__(self, memo):
        """Explicit deepcopy that skips the client and recreates it."""
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k == "client":
                continue
            setattr(new, k, copy.deepcopy(v, memo))
        # finally restore client
        new.client = openai.OpenAI(**new._client_kwargs)
        return new


class Gemini_LLM(LLM):
    """
    A manager class for handling requests to Google's Gemini models.
    """

    def __init__(self, api_key, model="gemini-2.0-flash", **kwargs):
        """
        Initializes the LLM manager with an API key and model name.

        Args:
            api_key (str): api key for authentication.
            model (str, optional): model abbreviation. Defaults to "gemini-2.0-flash".
                Options are: "gemini-1.5-flash","gemini-2.0-flash", and others from Googles models library.
        """
        super().__init__(api_key, model, None, **kwargs)
        genai.configure(api_key=api_key)
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        self.client = genai.GenerativeModel(
            model_name=self.model,  # "gemini-1.5-flash","gemini-2.0-flash",
            generation_config=generation_config,
            system_instruction="You are a computer scientist and excellent Python programmer.",
        )

    def _query(self, session_messages, max_retries: int = 5, default_delay: int = 10):
        """
        Sends the conversation history to Gemini, retrying on 429 ResourceExhausted exceptions.

        Args:
            session_messages (list[dict]): [{"role": str, "content": str}, …]
            max_retries (int): how many times to retry before giving up.
            default_delay (int): fallback sleep when the error has no retry_delay.

        Returns:
            str: model's reply.
        """
        history = [
            {"role": m["role"], "parts": [m["content"]]} for m in session_messages[:-1]
        ]
        last = session_messages[-1]["content"]

        attempt = 0
        while True:
            try:
                chat = self.client.start_chat(history=history)
                response = chat.send_message(last)
                return response.text

            except Exception as err:
                attempt += 1
                if attempt > max_retries:
                    raise  # bubble out after N tries

                # Prefer the structured retry_delay field if present
                delay = getattr(err, "retry_delay", None)
                if delay is not None:
                    wait = delay.seconds + 1  # add 1 second to avoid immediate retry
                else:
                    # Sometimes retry_delay only appears in the string—grab it
                    m = re.search(r"retry_delay\s*{\s*seconds:\s*(\d+)", str(err))
                    wait = int(m.group(1)) if m else default_delay * attempt

                time.sleep(wait)


class Ollama_LLM(LLM):
    def __init__(self, model="llama3.2", **kwargs):
        """
        Initializes the Ollama LLM manager with a model name. See https://ollama.com/search for models.

        Args:
            model (str, optional): model abbreviation. Defaults to "llama3.2".
                See for options: https://ollama.com/search.
        """
        super().__init__("", model, None, **kwargs)

    def _query(self, session_messages, max_retries: int = 5, default_delay: int = 10):
        """
        Sends a conversation history to the configured model and returns the response text.

        Args:
            session_messages (list of dict): A list of message dictionaries with keys
                "role" (e.g. "user", "assistant") and "content" (the message text).

        Returns:
            str: The text content of the LLM's response.
        """
        # first concatenate the session messages
        big_message = ""
        for msg in session_messages:
            big_message += msg["content"] + "\n"

        attempt = 0
        while True:
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": big_message}],
                )
                return response["message"]["content"]

            except ollama.ResponseError as err:
                attempt += 1
                if attempt > max_retries or err.status_code not in (429, 500, 503):
                    raise
                time.sleep(default_delay * attempt)

            except Exception:
                attempt += 1
                if attempt > max_retries:
                    raise
                time.sleep(default_delay * attempt)
