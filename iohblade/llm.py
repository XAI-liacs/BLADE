"""
LLM modules to connect to different LLM providers. Also extracts code, name and description.
"""

import copy
import logging
import re
import time
import random
from abc import ABC, abstractmethod
from typing import Any

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import ollama
except ImportError:
    ollama = None

try:
    import openai
except ImportError:
    openai = None

try:
    from google import genai
except ImportError:
    genai = None

try:
    import lmstudio as lms  # Platform dependent dependency.
except:
    lms = object

try:
    from mlx_lm import load, generate  # Platform dependent dependency.
except:
    load = None
    generate = None

try:
    from tokencost import (
        calculate_completion_cost,
        calculate_prompt_cost,
        count_message_tokens,
        count_string_tokens,
    )
except ImportError:
    calculate_completion_cost = None
    calculate_prompt_cost = None
    count_message_tokens = None
    count_string_tokens = None


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
    def _query(self, session: list, **kwargs):
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
        if (
            self.logger != None
            and hasattr(self.logger, "budget_exhausted")
            and self.logger.budget_exhausted()
        ):
            return "Budget exhausted."

        if self.log:
            input_msg = "\n".join([d["content"] for d in session])
            try:
                cost = calculate_prompt_cost(input_msg, self.model)
            except Exception:
                cost = 0
            try:
                tokens = count_message_tokens(input_msg, model=self.model)
            except Exception:
                tokens = 0
            self.logger.log_conversation(
                "client",
                input_msg,
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

    def sample_solution(
        self,
        session_messages: list,
        parent_ids=[],
        HPO=False,
        base_code: str | None = None,
        diff_mode: bool = False,
        **kwargs,
    ):
        """
        Interacts with a language model to generate or mutate solutions based on the provided session messages.

        Args:
            session_messages (list): A list of dictionaries with keys 'role' and 'content' to simulate a conversation with the language model.
            parent_ids (list, optional): The id of the parent the next sample will be generated from (if any).
            HPO (boolean, optional): If HPO is enabled, a configuration space will also be extracted (if possible).
            base_code and diff_mode are for now only there to support latest LLaMEA, they are not implemented yet.

        Returns:
            tuple: A tuple containing the new algorithm code, its class name, its full descriptive name and an optional configuration space object.
            **kwargs: Additional LLM settings that can be used at query time.

        Raises:
            NoCodeException: If the language model fails to return any code.
            Exception: Captures and logs any other exceptions that occur during the interaction.
        """
        message = self.query(session_messages, **kwargs)

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
        if ConfigurationSpace == None:
            raise Exception("Please install the ConfigSpace package first.")
        pattern = r"space\s*:\s*\n*```\n*(?:python)?\n(.*?)\n```"
        c = None
        for m in re.finditer(pattern, message, re.DOTALL | re.IGNORECASE):
            try:
                from ConfigSpace import ConfigurationSpace

                c = ConfigurationSpace(eval(m.group(1)))
            except Exception:
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


class Multi_LLM(LLM):
    def __init__(self, llms: list[LLM]):
        """
        Combine multiple LLM instances and randomly choose one per call.

        Args:
            llms (list[LLM]): A list of LLM instances to combine.
        """
        if not llms:
            raise ValueError("llms must contain at least one LLM instance")
        model = "multi-llm"
        super().__init__("", model)
        self.llms = llms

    def _pick_llm(self) -> LLM:
        """
        Randomly selects one of the LLMs from the list.
        This method is used to alternate between LLMs during evolution.
        """
        return random.choice(self.llms)

    def set_logger(self, logger):
        self.logger = logger
        self.log = True
        for llm in self.llms:
            llm.set_logger(logger)

    def _query(self, session_messages, **kwargs):
        llm = self._pick_llm()
        return llm._query(session_messages, **kwargs)


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

    def _query(
        self, session_messages, max_retries: int = 5, default_delay: int = 10, **kwargs
    ):
        """
        Sends a conversation history to the configured model and returns the response text.

        Args:
            session_messages (list of dict): A list of message dictionaries with keys
                "role" (e.g. "user", "assistant") and "content" (the message text).
            **kwargs: To add access to additional parameters that are avaiable
            in openai's client.chat.completions.create, we allow users to add
            their own parameters.

        Returns:
            str: The text content of the LLM's response.
        """

        attempt = 0
        while True:
            try:
                ## Manage temeperature copy.
                temperature = self.temperature
                if "temperature" in kwargs:
                    temperature = kwargs["temperature"]
                    kwargs.pop("temperature")

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=session_messages,
                    temperature=temperature,
                    **kwargs,
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


class DeepSeek_LLM(OpenAI_LLM):
    """A manager class for the DeepSeek chat models."""

    def __init__(self, api_key, model="deepseek-chat", temperature=0.8, **kwargs):
        """Initializes DeepSeek LLM with required base URL."""
        super().__init__(api_key, model=model, temperature=temperature, **kwargs)
        self.base_url = "https://api.deepseek.com"
        self._client_kwargs["base_url"] = self.base_url
        self.client = openai.OpenAI(**self._client_kwargs)


class Gemini_LLM(LLM):
    """
    A manager class for handling requests to Google's Gemini models.
    """

    def __init__(
        self, api_key, model="gemini-2.0-flash", generation_config=None, **kwargs
    ):
        """
        Initializes the LLM manager with an API key and model name.

        Args:
            api_key (str): api key for authentication.
            model (str, optional): model abbreviation. Defaults to "gemini-2.0-flash".
                Options are: "gemini-1.5-flash","gemini-2.0-flash", and others from Googles models library.
        """
        super().__init__(api_key, model, None, **kwargs)
        if generation_config is None:
            generation_config = {
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 65536,
                "response_mime_type": "text/plain",
            }

        self.client = genai.Client(api_key=api_key)
        self.api_key = api_key
        self.generation_config = generation_config

    def _query(
        self, session_messages, max_retries: int = 5, default_delay: int = 10, **kwargs
    ):
        """
        Sends the conversation history to Gemini, retrying on 429 ResourceExhausted exceptions.

        Args:
            session_messages (list[dict]): [{"role": str, "content": str}, …]
            max_retries (int): how many times to retry before giving up.
            default_delay (int): fallback sleep when the error has no retry_delay.
            kwargs: The generation_config is provided in __init__, to change a set
            of these config, or adding extra parameters, use kwargs, or
            additional named arguements to function.
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
                config = self.generation_config.copy()
                config.update(**kwargs)
                chat = self.client.chats.create(
                    model=self.model, history=history, config=config
                )
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

    # ---------- pickling / deepcopy helpers ----------
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("client", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.client = genai.Client(api_key=self.api_key)

    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k == "client":
                continue
            setattr(new, k, copy.deepcopy(v, memo))
        new.client = genai.Client(api_key=new.api_key)
        return new


class Ollama_LLM(LLM):
    def __init__(self, model="llama3.2", port=11434, **kwargs):
        """
        Initializes the Ollama LLM manager with a model name. See https://ollama.com/search for models.

        Args:
            model (str, optional): model abbreviation. Defaults to "llama3.2".
                See for options: https://ollama.com/search.
            port: TCP/UDP port on which localhost for ollama is available. Defaults to 11434.
        """
        self.port = port
        self.client = ollama.Client(host=f"http://localhost:{port}")

        super().__init__("", model, None, **kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("client", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.client = ollama.Client(host=f"http://localhost:{self.port}")

    def _query(
        self, session_messages, max_retries: int = 5, default_delay: int = 10, **kwargs
    ):
        """
        Sends a conversation history to the configured model and returns the response text.

        Args:
            session_messages (list of dict): A list of message dictionaries with keys
                "role" (e.g. "user", "assistant") and "content" (the message text).
            **kwargs Can be used to add additional `chat` parameters to the query,
                These queries are seperate queries placed under `option` parameter as
                documented in https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values


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
                response = self.client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": big_message}],
                    options=kwargs,
                )
                return response["message"]["content"]

            except ollama.ResponseError as err:
                attempt += 1
                if attempt > max_retries or err.status_code not in (429, 500, 503):
                    raise
                time.sleep(default_delay * attempt)


class Claude_LLM(LLM):
    """A manager class for handling requests to Anthropic's Claude models."""

    def __init__(
        self,
        api_key,
        model="claude-3-haiku-20240307",
        base_url=None,
        max_tokens=12000,
        temperature=0.8,
        **kwargs,
    ):
        """Initializes the LLM manager with an API key and model name."""

        super().__init__(api_key, model, base_url, **kwargs)
        self.temperature = temperature
        self._client_kwargs = {"api_key": api_key}
        self.max_tokens = max_tokens
        if base_url:
            self._client_kwargs["base_url"] = base_url
        self.client = anthropic.Anthropic(**self._client_kwargs)
        logging.getLogger("anthropic").setLevel(logging.ERROR)

    def _query(self, session_messages, max_retries: int = 5, default_delay: int = 10):
        """Sends a conversation history to the configured model and returns the response text."""

        attempt = 0
        while True:
            try:
                response = self.client.messages.create(
                    max_tokens=self.max_tokens,
                    model=self.model,
                    messages=session_messages,
                    temperature=self.temperature,
                )

                content = response.content
                if isinstance(content, list):
                    parts = []
                    for block in content:
                        if hasattr(block, "text"):
                            parts.append(block.text)
                        elif isinstance(block, dict) and "text" in block:
                            parts.append(block["text"])
                    text_output = "".join(parts)
                else:
                    text_output = str(content)
                return text_output

            except anthropic.RateLimitError as err:
                attempt += 1
                if attempt > max_retries:
                    raise
                retry_after = None
                if getattr(err, "response", None) is not None:
                    retry_after = err.response.headers.get("Retry-After")
                wait = int(retry_after) if retry_after else default_delay * attempt
                time.sleep(wait)

            except (
                anthropic.APITimeoutError,
                anthropic.APIConnectionError,
                anthropic.APIError,
            ) as err:
                attempt += 1
                if attempt > max_retries:
                    raise
                time.sleep(default_delay * attempt)

    # ---------- pickling / deepcopy helpers ----------
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("client", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.client = anthropic.Anthropic(**self._client_kwargs)

    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k == "client":
                continue
            setattr(new, k, copy.deepcopy(v, memo))
        new.client = anthropic.Anthropic(**new._client_kwargs)
        return new


class LMStudio_LLM(LLM):
    """A manager for running MLX-Optimised LLM locally."""

    def __init__(self, model, config=None, **kwargs):
        """
        Initialises the LMStudio LLM inteface.

        :param model: Name of the model, to be initialised for interaction.
        :param config: Configuration to be set for LLM chat.
        :param kwargs: Keyed arguements for setting up the LLM chat.
        """
        super().__init__(api_key="", model=model, **kwargs)
        self.llm = lms.llm(model)
        self.config = config

    def _query(self, session: list[dict[str, str]], max_tries: int = 5) -> str:
        """
        Query stub for LMStudio class.

        ## Parameters
        `session: list[dict[str, str]]`: A session message is a list of {'role' : 'user'|'system', 'content': 'content'} data, use to make LLM request.
        `max_tries: int`: A max count for the number of tries, to get a response.
        """
        request = session[-1]["content"]
        for _ in range(max_tries):
            try:
                if self.config is not None:
                    response = self.llm.respond(request, config=self.config)
                else:
                    response = self.llm.respond(request)
                response = re.sub(  # Remove thinking section, if avaiable.
                    r"<think>.*?</think>", "", str(response), flags=re.DOTALL
                )
                return response
            except:
                pass
        return ""

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("llm", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.llm = lms.llm(self.model)

    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k == "llm":
                continue
            setattr(new, k, copy.deepcopy(v, memo))
        new.llm = self.llm
        return new


class MLX_LM_LLM(LLM):
    """An mlx_lm implementation for running large LLMs locally."""

    def __init__(self, model, config=None, max_tokens: int = 12000, **kwargs):
        """
        Initialises the LMStudio LLM inteface.

        :param model: Name of the model, to be initialised for interaction.
        :param config: Configuration to be set for LLM chat.
        :param max_tokens: Maximun number of tokens to be generated for a request.
        :param kwargs: Keyed arguements for setting up the LLM chat.
        """
        super().__init__(api_key="", model=model, **kwargs)
        if config is not None:
            llm, tokenizer = load(model, model_config=config)
        else:
            llm, tokenizer = load(model)
        self.llm = llm
        self.tokenizer = tokenizer

        self.config = config
        self.max_tokens = max_tokens

    def __getstate__(self) -> object:
        state = self.__dict__
        state.pop("tokenizer", None)
        state.pop("llm", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.config is None:
            llm, tokenizer = load(self.model)
        else:
            llm, tokenizer = load(self.model, model_config=self.config)
        self.llm = llm
        self.tokenizer = tokenizer

    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k in ["llm", "tokenizer"]:
                continue
            setattr(new, k, copy.deepcopy(v, memo))
        new.llm = self.llm  # <- reference symantics copy for massive object `llm`.
        new.tokenizer = self.tokenizer
        return new

    def _query(
        self, session: list, max_tries: int = 5, add_generation_prompt: bool = False
    ):
        """
        Query stub for LMStudio class.

        ## Parameters
        `session: list[dict[str, str]]`: A session message is a list of {'role' : 'user'|'system', 'content': 'content'} data, use to make LLM request.
        `max_tries: int`: A max count for the number of tries, to get a response.
        `add_generation_prompt: bool`: MLX_LM come with an option to add_generation_prompt to optimise prompts.
        """
        prompt = self.tokenizer.apply_chat_template(
            session, add_generation_prompt=add_generation_prompt
        )
        for _ in range(max_tries):
            try:
                response = generate(
                    self.llm,
                    self.tokenizer,
                    prompt,
                    max_tokens=self.max_tokens,  # Disable limit on token count.
                )
                response = re.sub(  # Remove thinking section, if avaiable.
                    r"<think>.*?</think>", "", str(response), flags=re.DOTALL
                )
                return response
            except:
                pass
        return ""


class Dummy_LLM(LLM):
    def __init__(self, model="DUMMY", **kwargs):
        """
        Initializes the DUMMY LLM manager with a model name. This is a placeholder
        and does not connect to any LLM provider. It is used for testing purposes only.

        Args:
            model (str, optional): model abbreviation. Defaults to "DUMMY".
                Has no effect, just a placeholder.
        """
        super().__init__("", model, None, **kwargs)

    def _query(self, session_messages):
        """
        Sends a conversation history to DUMMY model and returns a random response text.

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
        response = """This is a dummy response from the DUMMY LLM. It does not connect to any LLM provider.
It is used for testing purposes only.
# Description: A simple random search algorithm that samples points uniformly in the search space and returns the best found solution.
# Code:
```python
import numpy as np

class RandomSearch:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        for i in range(self.budget):
            x = np.random.uniform(func.bounds.lb, func.bounds.ub)
            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x
        return self.f_opt, self.x_opt
```
# Configuration Space:
```python
{
    'budget': {'type': 'int', 'lower': 1000, 'upper': 100000},
    'dim': {'type': 'int', 'lower': 1, 'upper': 100}
}
```
"""
        return response
