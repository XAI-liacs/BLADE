"""
LLM modules to connect to different LLM providers. Also extracts code, name and description.
"""

import copy
import logging
import random
import re
import threading
import time
import uuid
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

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
    from mlx_lm import generate, load  # Platform dependent dependency.
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


import platform
import subprocess

import psutil
from cpuinfo import get_cpu_info

from .solution import Solution
from .utils import NoCodeException


def _run(cmd):
    try:
        return subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return ""


def _get_gpu_info() -> tuple[Optional[str], Optional[int]]:
    system = platform.system()

    # WINDOWS
    if system == "Windows":
        try:
            output = _run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    (
                        "Get-CimInstance Win32_VideoController | "
                        "Select-Object Name,AdapterRAM | ConvertTo-Csv -NoTypeInformation"
                    ),
                ]
            )

            lines = [x for x in output.splitlines() if x.strip()]
            if len(lines) >= 2:
                cols = [c.strip('"') for c in lines[1].split(",")]

                gpu_name = cols[0]

                video_memory = None
                if len(cols) > 1 and cols[1].isdigit():
                    video_memory = int(cols[1]) // (1024**2)

                return gpu_name, video_memory

        except Exception:
            pass

    # LINUX
    elif system == "Linux":
        try:
            output = _run(["lspci"])

            gpu_lines = [
                line
                for line in output.splitlines()
                if (
                    # "VGA compatible controller" in line //Not the best predictor of LLM GPU.
                    "3D controller" in line
                    or "Display controller" in line
                )
            ]

            if gpu_lines:
                gpu_name = gpu_lines[0].split(": ", 1)[1]

                # NVIDIA VRAM
                try:
                    smi = _run(
                        [
                            "nvidia-smi",
                            "--query-gpu=memory.total",
                            "--format=csv,noheader,nounits",
                        ]
                    )
                    if smi:
                        return gpu_name, int(smi.splitlines()[0])
                except Exception:
                    pass

                return gpu_name, None

        except Exception:
            pass

    # MACOS
    elif system == "Darwin":
        try:
            output = _run(["system_profiler", "SPDisplaysDataType"])

            gpu_name = None
            vram = None

            for line in output.splitlines():
                line = line.strip()

                if line.startswith("Chipset Model:"):
                    gpu_name = line.split(":", 1)[1].strip()

                elif line.startswith("VRAM"):
                    m = re.search(r"(\d+)", line)
                    if m:
                        vram = int(m.group(1))

            return gpu_name, vram

        except Exception:
            pass

    return None, None


def get_hardware_info() -> Dict:
    cpu_name = None

    try:
        cpu_name = get_cpu_info().get("brand_raw")
    except Exception:
        cpu_name = platform.processor()

    gpu_name, video_memory = _get_gpu_info()

    memory_gb = round(psutil.virtual_memory().total / (1024**2), 1)

    return {
        "cpu_name": cpu_name,
        "gpu_name": gpu_name,
        "memory": memory_gb,  # MB
        "video_memory": video_memory,  # MB
        "cpu_core_count": psutil.cpu_count(logical=False),
        "os_name": platform.platform(),
    }


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
                "class\\s*(\\w*)(?:\\([^)]*\\))?\\:",
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
        try:
            from ConfigSpace import ConfigurationSpace
        except ImportError:
            # ConfigSpace not installed, no HPO
            return None

        pattern = r"space\s*:\s*\n*```\n*(?:python)?\n(.*?)\n```"
        c = None
        for m in re.finditer(pattern, message, re.DOTALL | re.IGNORECASE):
            try:
                cfg_dict = eval(m.group(1), {"__builtins__": {}})
                c = ConfigurationSpace(cfg_dict)
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

    @abstractmethod
    def get_config(self) -> list[dict[str, Any]]:
        """
        Return the configuration of the llms as a list of dictionary, of the format:
        [
            {
                "model": str,
                "config": dict,
                "hardware": dict *only available when hardware details are known.
            },
            ...,
        ]
        ## Params:
        None

        ## Returns:
        list[dict['str': Any]]
        """

        pass


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

    def get_config(self) -> list[dict[str, Any]]:
        config_data = []
        for llm in self.llms:
            config_data.extend(llm.get_config())
        return config_data


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

    def get_config(self) -> list[dict[str, Any]]:
        llm_config = self._client_kwargs.copy()
        llm_config["temperature"] = self.temperature
        llm_config.pop("api_key")
        config = {"model": f"{self.model}", "config": llm_config, "hardware": {}}
        return [config]

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

    def get_config(self) -> list[dict[str, Any]]:
        llm_config = self._client_kwargs.copy()
        llm_config.pop("api_key")
        llm_config["temperature"] = self.temperature
        config = {"model": f"{self.model}", "config": llm_config, "hardware": {}}
        return [config]


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

    def get_config(self) -> list[dict[str, Any]]:
        config = {"model": self.model, "config": self.generation_config, "hardware": {}}
        return [config]


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

    def get_config(self) -> list[dict[str, Any]]:
        hw_info = get_hardware_info()
        config = {
            "model": self.model,
            "config": {"port": self.port},
            "hardware": hw_info,
        }
        return [config]


class Claude_LLM(LLM):
    """A manager class for handling requests to Anthropic's Claude models."""

    def __init__(
        self,
        api_key,
        model="claude-opus-5-5",
        base_url=None,
        max_tokens=16000,
        temperature=None,
        effort=None,
        fallbacks=None,
        **kwargs,
    ):
        """Initializes the LLM manager with an API key and model name.

        Args:
            api_key (str | None): Anthropic API key. ``None`` lets the SDK
                resolve credentials (``ANTHROPIC_API_KEY``, an ``ant auth
                login`` profile, ...).
            model (str): Claude model ID.
            base_url (str, optional): Alternative API base URL.
            max_tokens (int): Output token cap per response (thinking
                included).
            temperature (float | None): Deprecated and ignored. The Claude
                API (version 2026-10-21) no longer accepts sampling
                parameters; requests use the API's default sampling.
            effort (str | None): ``output_config.effort`` (``"low"`` to
                ``"max"``); ``None`` uses the model's default.
            fallbacks (str | list | None): Server-side refusal fallbacks,
                e.g. ``"default"``. Sent through the beta endpoint.
        """

        super().__init__(api_key, model, base_url, **kwargs)
        if temperature is not None:
            warnings.warn(
                "Claude_LLM ignores temperature: the Claude API no longer "
                "accepts sampling parameters.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.temperature = temperature
        self.effort = effort
        self.fallbacks = fallbacks
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
                request = {
                    "max_tokens": self.max_tokens,
                    "model": self.model,
                    "messages": session_messages,
                }
                if self.effort is not None:
                    request["output_config"] = {"effort": self.effort}
                if self.fallbacks is not None:
                    fallback_beta = (
                        "server-side-fallback-2026-07-01"
                        if isinstance(self.fallbacks, str)
                        else "server-side-fallback-2026-06-01"
                    )
                    response = self.client.beta.messages.create(
                        betas=[fallback_beta], fallbacks=self.fallbacks, **request
                    )
                else:
                    response = self.client.messages.create(**request)

                if response.stop_reason == "refusal":
                    # Declined (by the fallback chain too, if one was set):
                    # any partial output is not a usable answer.
                    return ""
                content = response.content
                if isinstance(content, list):
                    parts = []
                    for block in content:
                        if getattr(block, "type", None) == "text":
                            parts.append(block.text)
                        elif (
                            isinstance(block, dict)
                            and block.get("type", "text") == "text"
                            and "text" in block
                        ):
                            parts.append(block["text"])
                    text_output = "".join(parts)
                else:
                    text_output = str(content)
                return text_output

            except (
                anthropic.BadRequestError,
                anthropic.AuthenticationError,
                anthropic.PermissionDeniedError,
                anthropic.NotFoundError,
            ):
                raise  # retrying cannot fix these

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
        state.setdefault("effort", None)  # older pickles lack these
        state.setdefault("fallbacks", None)
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

    def get_config(self) -> list[dict[str, Any]]:
        # Never include the API key: configs are written to experiment logs.
        llm_config = {k: v for k, v in self._client_kwargs.items() if k != "api_key"}
        llm_config["max_tokens"] = self.max_tokens
        llm_config["effort"] = self.effort
        llm_config["fallbacks"] = self.fallbacks

        config = {"model": self.model, "config": llm_config}
        return [config]


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
        self.llm = object
        if lms is not object:
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
                response = (
                    self.llm.respond(request, config=self.config)
                    if self.config is not None
                    else self.llm.respond(request)
                )

                text = "".join(str(chunk) for chunk in response)
                response = re.sub(  # Remove thinking section, if avaiable.
                    r"<think>.*?</think>", "", str(text), flags=re.DOTALL
                )
                return response
            except:
                time.sleep(0.2)
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

    def get_config(self) -> list[dict[str, Any]]:
        hw_info = get_hardware_info()
        config = {"model": self.model, "config": self.config or {}, "hardware": hw_info}
        return [config]


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

    def get_config(self) -> list[dict[str, Any]]:
        hw_info = get_hardware_info()
        config = {"model": self.model, "config": self.config or {}, "hardware": hw_info}
        return [config]


def _find_decoder_layers(model) -> list:
    """Returns the list of transformer blocks of a HuggingFace causal LM.

    Covers the common layouts (Llama/Qwen/Gemma/Mistral: ``model.model.layers``,
    GPT-2/GPT-J: ``model.transformer.h``, GPT-NeoX: ``model.gpt_neox.layers``)
    and multimodal wrappers that nest a language model one level deeper.

    Raises:
        ValueError: If no list of blocks can be found.
    """
    candidates = [
        ("model", "layers"),
        ("model", "language_model", "layers"),
        ("language_model", "model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for path in candidates:
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return list(obj)
    raise ValueError(
        f"Could not locate the decoder layers of {type(model).__name__}; "
        "pass a supported HuggingFace causal LM."
    )


class HookedLLM(LLM):
    """Open-weights HuggingFace model with activation capture and steering.

    Drop-in provider for every BLADE method (LLaMEA, RandomSearch, ...) that
    additionally records the model's residual stream while it designs
    algorithms, for mechanistic-interpretability experiments.

    For every query, the following per-layer summaries of the residual stream
    (output of each decoder block) are captured during generation:

    * ``last``: the final prompt token, i.e. the state right before the model
      starts writing. This is the standard position for probing "intent".
    * ``prompt_mean``: mean over all prompt tokens.
    * ``gen_mean``: mean over the generated tokens (the state while writing
      the code).

    Plus cheap output-uncertainty baselines: mean token entropy and mean
    log-probability of the sampled tokens, computed on the model's own
    temperature-1 distribution (before temperature and top-p are applied).

    Records are written to ``<run_dir>/mi_activations/`` when a BLADE run
    logger is attached (one ``.npz`` per query and a line in ``index.jsonl``
    that links the record to ``solution.id``), else to ``activation_dir``,
    else kept in memory in ``self.records``.

    Full per-token activations are not stored during the run (too large);
    use :meth:`replay` to recompute them offline from the logged prompt and
    response.

    Requires the optional ``mi`` dependency group (``torch``,
    ``transformers``, ``accelerate``).
    """

    ARRAY_KEYS = ("last", "prompt_mean", "gen_mean")

    def __init__(
        self,
        model: str,
        capture_layers: list[int] | None = None,
        capture: bool = True,
        activation_dir: str | None = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.8,
        top_p: float = 0.95,
        dtype: str = "bfloat16",
        device_map: str | dict = "auto",
        chat_template_kwargs: dict | None = None,
        store_text: bool = True,
        store_dtype: str = "float16",
        hf_model=None,
        tokenizer=None,
        **kwargs,
    ):
        """Loads the model and registers the hooks.

        Args:
            model: HuggingFace model id or local path.
            capture_layers: Decoder layer indices to record. ``None`` records
                all layers (needed for layer-wise probing curves).
            capture: Whether to record activations at all.
            activation_dir: Where to store records when no run logger is
                attached. ``None`` keeps them in memory.
            max_new_tokens: Generation limit per query.
            temperature: Sampling temperature (0 means greedy decoding).
            top_p: Nucleus sampling threshold.
            dtype: Torch dtype name for the weights.
            device_map: Passed to ``from_pretrained``, e.g. ``"auto"`` or
                ``"cuda:1"``.
            chat_template_kwargs: Extra arguments for the chat template,
                e.g. ``{"enable_thinking": False}`` for Qwen3.
            store_text: Store prompt and response text in ``index.jsonl``
                so a record is self-contained for offline replay.
            store_dtype: NumPy dtype for stored activations. ``"float16"``
                halves storage but overflows for models whose residual
                stream exceeds 65504 (e.g. Gemma 3); use ``"float32"`` then.
            hf_model: Optional pre-loaded model (skips loading ``model``).
            tokenizer: Optional pre-loaded tokenizer.
            **kwargs: Passed to :class:`LLM` (regex patterns, logger).
        """
        super().__init__(api_key="", model=model, **kwargs)
        self.capture_layers = capture_layers
        self.capture = capture
        self.activation_dir = activation_dir
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.dtype = dtype
        self.device_map = device_map
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.store_text = store_text
        self.store_dtype = store_dtype
        self.records: list[dict] = []
        self.last_record: dict | None = None
        self._pending: dict | None = None
        self._in_sample = False
        self._query_count = 0
        self._steering: dict[int, Any] = {}
        self._steering_alpha = 1.0
        self._steering_positions = "generation"
        self._state: dict = {}
        self._lock = threading.Lock()
        self._load(hf_model, tokenizer)

    # ------------------------------------------------------------------
    # Model loading, (de)serialisation and copying
    # ------------------------------------------------------------------
    def _load(self, hf_model=None, tokenizer=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.model)
        if hf_model is None:
            hf_model = AutoModelForCausalLM.from_pretrained(
                self.model,
                dtype=getattr(torch, self.dtype),
                device_map=self.device_map,
            )
        hf_model.eval()
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.layers = _find_decoder_layers(hf_model)
        self.n_layers = len(self.layers)
        if self.capture_layers is None:
            self.capture_layers = list(range(self.n_layers))
        self._hook_handles = [
            layer.register_forward_hook(self._make_hook(i))
            for i, layer in enumerate(self.layers)
        ]

    _HEAVY = ("hf_model", "tokenizer", "layers", "_hook_handles", "_lock")

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in self._HEAVY:
            state.pop(key, None)
        state["_state"] = {}
        state["_steering"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()
        self._load()

    def __deepcopy__(self, memo):
        # BLADE deep-copies the method (and its LLM) for every run. Share the
        # weights, the hooks and the GPU lock instead of reloading the model.
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k in self._HEAVY or k in ("_state", "_steering"):
                continue
            setattr(new, k, copy.deepcopy(v, memo))
        for k in self._HEAVY:
            setattr(new, k, getattr(self, k))
        new._state = {}
        new._steering = dict(self._steering)
        # The hooks dispatch to whichever instance is generating, see
        # ``_make_hook``, so copies need no hooks of their own.
        return new

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    # Hooks are registered once on the shared model and dispatch to the
    # instance that is currently generating (set under the lock).
    _active_by_model: dict[int, "HookedLLM"] = {}

    def _make_hook(self, layer_idx: int):
        model_key = id(self.hf_model)

        def hook(module, inputs, output):
            owner = HookedLLM._active_by_model.get(model_key)
            if owner is None:
                return None
            return owner._on_layer(layer_idx, output)

        return hook

    def _on_layer(self, layer_idx: int, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        st = self._state
        prefill = hs.shape[1] > 1 or not st.get("prefilled", False)

        vec = self._steering.get(layer_idx)
        if vec is not None:
            add = self._steering_alpha * vec.to(hs.device, hs.dtype)
            split = st.get("replay_split")
            if self._steering_positions == "all":
                hs = hs + add
            elif split is not None:
                # Teacher-forced replay: only the response was steered.
                hs = hs.clone()
                hs[:, split:] += add
            elif not prefill:
                hs = hs + add

        if "full" in st and layer_idx in st["replay_layers"]:
            st["full"][layer_idx] = hs[0].detach().float().cpu()
        if st.get("capture") and layer_idx in st["layers"]:
            h = hs[0].detach().float()
            if prefill:
                st["last"][layer_idx] = h[-1].cpu()
                st["prompt_mean"][layer_idx] = h.mean(0).cpu()
            else:
                acc = st["gen_sum"].get(layer_idx)
                st["gen_sum"][layer_idx] = h[-1] if acc is None else acc + h[-1]
                st["gen_count"][layer_idx] = st["gen_count"].get(layer_idx, 0) + 1
        if layer_idx == self.n_layers - 1:
            st["prefilled"] = True

        if vec is None:
            return None
        return (hs,) + tuple(output[1:]) if is_tuple else hs

    # ------------------------------------------------------------------
    # Steering (experiment 5)
    # ------------------------------------------------------------------
    def set_steering(
        self,
        vectors: dict,
        alpha: float = 1.0,
        positions: str = "generation",
    ):
        """Adds ``alpha * vector`` to the residual stream at given layers.

        Args:
            vectors: Mapping ``layer index -> vector`` (numpy or torch,
                shape ``[d_model]``).
            alpha: Global scaling factor.
            positions: ``"generation"`` steers only generated tokens,
                ``"all"`` also steers the prompt.
        """
        import torch

        if positions not in ("generation", "all"):
            raise ValueError("positions must be 'generation' or 'all'")
        self._steering = {
            int(k): torch.as_tensor(v).flatten() for k, v in vectors.items()
        }
        self._steering_alpha = float(alpha)
        self._steering_positions = positions

    def clear_steering(self):
        """Removes all steering vectors."""
        self._steering = {}

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------
    def _encode_chat(self, session: list[dict]):
        """Applies the chat template; merges a system turn if unsupported."""
        try:
            return self.tokenizer.apply_chat_template(
                session,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                **self.chat_template_kwargs,
            )
        except Exception:
            merged = [dict(m) for m in session]
            if merged and merged[0]["role"] == "system" and len(merged) > 1:
                system = merged.pop(0)["content"]
                merged[0]["content"] = system + "\n\n" + merged[0]["content"]
            return self.tokenizer.apply_chat_template(
                merged,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                **self.chat_template_kwargs,
            )

    def _entropy_processor(self, stats: dict):
        import torch
        from transformers import LogitsProcessor

        class _Stats(LogitsProcessor):
            def __call__(self, input_ids, scores):
                logp = torch.log_softmax(scores[0].float(), dim=-1)
                prev = stats.get("prev_logp")
                if prev is not None:
                    stats["logprob_sum"] += float(prev[input_ids[0, -1]])
                    stats["n_logprob"] += 1
                finite = torch.isfinite(logp)
                p = logp.exp()
                ent = -(p[finite] * logp[finite]).sum()
                stats["entropy_sum"] += float(ent)
                stats["n_entropy"] += 1
                stats["prev_logp"] = logp
                return scores

        return _Stats()

    def _query(self, session: list, **kwargs):
        import numpy as np
        import torch

        enc = self._encode_chat(session)
        device = self.hf_model.get_input_embeddings().weight.device
        enc = {k: v.to(device) for k, v in enc.items()}
        n_prompt = int(enc["input_ids"].shape[1])
        stats = {"entropy_sum": 0.0, "n_entropy": 0, "logprob_sum": 0.0}
        stats["n_logprob"] = 0
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            logits_processor=[self._entropy_processor(stats)],
        )
        if self.temperature and self.temperature > 0:
            gen_kwargs.update(
                do_sample=True, temperature=self.temperature, top_p=self.top_p
            )
        else:
            gen_kwargs.update(do_sample=False)

        with self._lock:
            # Seed torch from numpy's global RNG, which BLADE seeds per run,
            # so runs are reproducible and different seeds sample differently.
            torch.manual_seed(int(np.random.randint(0, 2**31 - 1)))
            self._state = {
                "capture": self.capture,
                "layers": set(self.capture_layers),
                "last": {},
                "prompt_mean": {},
                "gen_sum": {},
                "gen_count": {},
                "prefilled": False,
            }
            HookedLLM._active_by_model[id(self.hf_model)] = self
            start = time.time()
            try:
                with torch.no_grad():
                    out = self.hf_model.generate(**enc, **gen_kwargs)
            finally:
                HookedLLM._active_by_model.pop(id(self.hf_model), None)
            elapsed = time.time() - start
            st, self._state = self._state, {}

        gen_ids = out[0, n_prompt:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        self._query_count += 1
        record = {
            "record_id": uuid.uuid4().hex,
            "query_idx": self._query_count - 1,
            "model": self.model,
            "n_prompt_tokens": n_prompt,
            "n_gen_tokens": int(gen_ids.shape[0]),
            "mean_entropy": stats["entropy_sum"] / max(stats["n_entropy"], 1),
            "mean_logprob": stats["logprob_sum"] / max(stats["n_logprob"], 1),
            "gen_time_s": elapsed,
            "steering": self._steering_description(),
            "layers": sorted(st["last"].keys()) if self.capture else [],
        }
        if self.store_text:
            record["prompt"] = session
            record["response"] = text
        if self.capture:
            record["arrays"] = self._stack_arrays(st)
        self._pending = record
        return text

    def _stack_arrays(self, st: dict) -> dict:
        import numpy as np

        layers = sorted(st["last"].keys())
        arrays = {}
        if not layers:
            return arrays
        arrays["last"] = np.stack([st["last"][l].numpy() for l in layers])
        arrays["prompt_mean"] = np.stack([st["prompt_mean"][l].numpy() for l in layers])
        d = arrays["last"].shape[1]
        gen = []
        for l in layers:
            if st["gen_count"].get(l):
                gen.append((st["gen_sum"][l] / st["gen_count"][l]).cpu().numpy())
            else:
                gen.append(np.full(d, np.nan, dtype=np.float32))
        arrays["gen_mean"] = np.stack(gen)
        dtype = getattr(self, "store_dtype", "float16")
        return {k: v.astype(dtype) for k, v in arrays.items()}

    def _steering_description(self):
        if not self._steering:
            return None
        return {
            "layers": sorted(self._steering.keys()),
            "alpha": self._steering_alpha,
            "positions": self._steering_positions,
            "norms": {
                str(k): float(v.float().norm()) for k, v in self._steering.items()
            },
        }

    def query(self, session: list, **kwargs):
        message = super().query(session)
        if not self._in_sample and self._pending is not None:
            self._flush(solution_id=None, parent_ids=[])
        return message

    def sample_solution(
        self,
        session_messages: list,
        parent_ids=[],
        HPO=False,
        base_code: str | None = None,
        diff_mode: bool = False,
        **kwargs,
    ):
        """Generates a solution and links the captured activations to it."""
        self._in_sample = True
        self._pending = None
        try:
            solution = super().sample_solution(
                session_messages,
                parent_ids,
                HPO=HPO,
                base_code=base_code,
                diff_mode=diff_mode,
                **kwargs,
            )
        finally:
            self._in_sample = False
        if self._pending is not None:
            record_id = self._flush(solution.id, list(parent_ids))
            solution.add_metadata("mi_record_id", record_id)
        return solution

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    def _record_dir(self) -> str | None:
        import os

        run_dir = getattr(self.logger, "dirname", None) if self.logger else None
        if run_dir:
            return os.path.join(run_dir, "mi_activations")
        return self.activation_dir

    def _flush(self, solution_id, parent_ids) -> str:
        import json
        import os

        import numpy as np

        record, self._pending = self._pending, None
        record["solution_id"] = solution_id
        record["parent_ids"] = parent_ids
        self.last_record = record
        out_dir = self._record_dir()
        if out_dir is None:
            self.records.append(record)
            return record["record_id"]
        os.makedirs(out_dir, exist_ok=True)
        arrays = record.pop("arrays", {})
        if arrays:
            fname = f"{record['record_id']}.npz"
            np.savez_compressed(os.path.join(out_dir, fname), **arrays)
            record["file"] = fname
        with open(os.path.join(out_dir, "index.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")
        record["arrays"] = arrays
        return record["record_id"]

    # ------------------------------------------------------------------
    # Offline analysis
    # ------------------------------------------------------------------
    def replay(
        self,
        session: list[dict],
        response: str,
        layers: list[int] | None = None,
    ) -> dict:
        """Teacher-forces ``prompt + response`` and returns per-token states.

        Use this to recompute full activations for a logged query (e.g. for
        attribution or SAE analysis) without storing them during the run.
        Steering vectors, if set, are applied as during generation (with
        ``positions="generation"`` only to the response tokens).

        Args:
            session: The chat messages of the query.
            response: The model's response text.
            layers: Layers to return (default: ``capture_layers``).

        Returns:
            dict: ``hidden`` (``store_dtype`` array ``[n_layers, n_tokens, d]``),
            ``layers``, ``tokens`` (strings), ``input_ids`` and
            ``n_prompt_tokens`` (prompt/response boundary).
        """
        import numpy as np
        import torch

        layers = self.capture_layers if layers is None else layers
        enc = self._encode_chat(session)
        prompt_ids = enc["input_ids"]
        resp_ids = self.tokenizer(
            response, add_special_tokens=False, return_tensors="pt"
        )["input_ids"]
        ids = torch.cat([prompt_ids, resp_ids], dim=1)
        device = self.hf_model.get_input_embeddings().weight.device
        with self._lock:
            HookedLLM._active_by_model[id(self.hf_model)] = self
            self._state = {
                "capture": False,
                "prefilled": False,
                "replay_split": int(prompt_ids.shape[1]),
                "replay_layers": set(layers),
                "full": {},
            }
            try:
                with torch.no_grad():
                    self.hf_model(input_ids=ids.to(device))
                full = self._state["full"]
            finally:
                HookedLLM._active_by_model.pop(id(self.hf_model), None)
                self._state = {}
        # Raw block outputs (HF's output_hidden_states applies the final norm
        # to the last layer, which would not match the captured records).
        dtype = getattr(self, "store_dtype", "float16")
        hidden = np.stack([full[l].numpy().astype(dtype) for l in layers])
        ids_list = ids[0].tolist()
        return {
            "hidden": hidden,
            "layers": list(layers),
            "input_ids": ids_list,
            "tokens": self.tokenizer.convert_ids_to_tokens(ids_list),
            "n_prompt_tokens": int(prompt_ids.shape[1]),
        }

    def to_dict(self):
        d = super().to_dict()
        d.update(self._settings())
        return d

    def _settings(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "dtype": self.dtype,
            "capture": self.capture,
            "capture_layers": self.capture_layers,
            "chat_template_kwargs": self.chat_template_kwargs,
        }

    def get_config(self) -> list[dict[str, Any]]:
        return [
            {
                "model": self.model,
                "config": self._settings(),
                "hardware": get_hardware_info(),
            }
        ]


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

    def get_config(self) -> list[dict[str, Any]]:
        return [{"model": self.model, "config": {}, "hardware": {}}]
