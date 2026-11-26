"""
LLM modules to connect to different LLM providers. Also extracts code, name and description.
"""

import copy
import importlib
import logging
import random
import re
import time
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


_DEFAULT_CODE_PATTERN = r"```(?:python)?\n(.*?)\n```"
_DEFAULT_NAME_PATTERN = "class\\s*(\\w*)(?:\\(\\w*\\))?\\:"
_DEFAULT_DESC_PATTERN = r"#\s*Description\s*:\s*(.*)"
_DEFAULT_CS_PATTERN = r"space\s*:\s*\n*```\n*(?:python)?\n(.*?)\n```"

# A fenced block: an opening fence at the start of a line with an optional
# language tag, the body, and a closing fence on its own line.
_FENCED_BLOCK = re.compile(
    r"^[ \t]*```[ \t]*([\w+.-]*)[^\n]*\n(.*?)\n[ \t]*```",
    re.MULTILINE | re.DOTALL,
)
_PYTHON_TAGS = ("python", "python3", "py")


def _extract_python_block(message: str) -> str | None:
    """The code of the first python-tagged fenced block, else of the first
    untagged one; blocks tagged with another language are skipped.

    Handles ```py, trailing spaces after the tag and CRLF line endings.
    """
    text = message.replace("\r\n", "\n")
    untagged = None
    for match in _FENCED_BLOCK.finditer(text):
        tag = match.group(1).lower()
        if tag in _PYTHON_TAGS:
            return match.group(2)
        if not tag and untagged is None:
            untagged = match.group(2)
    if untagged is not None:
        return untagged
    # Fences that do not start a line (e.g. "Code: ```python ...").
    match = re.search(_DEFAULT_CODE_PATTERN, text, re.DOTALL | re.IGNORECASE)
    return match.group(1) if match else None


def _main_class_name(code: str) -> str:
    """The name of the class an algorithm is instantiated from.

    Of the top-level classes, the one no other top-level class refers to
    (helpers such as ``Particle`` are used by the main class); if several,
    the last one. Falls back to the first ``class Name`` line when the code
    does not parse. Returns "" when there is no class.
    """
    import ast

    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        match = re.search(r"^[ \t]*class[ \t]+(\w+)", code, re.MULTILINE)
        return match.group(1) if match else ""
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if not classes:
        return ""
    used = set()
    for cls in classes:
        for node in ast.walk(cls):
            if isinstance(node, ast.Name) and node.id != cls.name:
                used.add(node.id)
    candidates = [cls.name for cls in classes if cls.name not in used]
    return (candidates or [cls.name for cls in classes])[-1]


def _retry_after_seconds(err, fallback: float) -> float:
    """Seconds from an error's Retry-After header, or ``fallback``."""
    response = getattr(err, "response", None)
    value = None
    if response is not None:
        try:
            value = response.headers.get("Retry-After")
        except Exception:
            value = None
    try:
        return max(0.0, float(value)) if value is not None else fallback
    except (TypeError, ValueError):  # e.g. an HTTP date
        return fallback


def _retryable_status(status) -> bool:
    """Whether an HTTP status is worth retrying (rate limit, server error)."""
    return status is None or status in (408, 409, 429) or status >= 500


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
        # Custom patterns replace the default extraction; the defaults are
        # stored so that to_dict() shows what is used.
        self.code_pattern = (
            code_pattern if code_pattern is not None else _DEFAULT_CODE_PATTERN
        )
        self.name_pattern = (
            name_pattern if name_pattern is not None else _DEFAULT_NAME_PATTERN
        )
        self.desc_pattern = (
            desc_pattern if desc_pattern is not None else _DEFAULT_DESC_PATTERN
        )
        self.cs_pattern = cs_pattern if cs_pattern is not None else _DEFAULT_CS_PATTERN
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

    def query(self, session: list, **kwargs):
        """
        Sends a conversation history to the configured model and returns the response text.

        Args:
            session_messages (list of dict): A list of message dictionaries with keys
                "role" (e.g. "user", "assistant") and "content" (the message text).
            **kwargs: Settings passed on to the provider's ``_query`` (for
                example ``temperature``), for providers that accept them.

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
                tokens = count_string_tokens(prompt=input_msg, model=self.model)
            except Exception:
                tokens = 0
            self.logger.log_conversation(
                "client",
                input_msg,
                cost,
                tokens,
            )

        message = self._query(session, **kwargs) if kwargs else self._query(session)

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
        if diff_mode and not getattr(LLM, "_warned_diff_mode", False):
            LLM._warned_diff_mode = True
            warnings.warn(
                "diff_mode is not supported by BLADE's LLM classes: responses "
                "are treated as complete code. Use diff_mode=False.",
                stacklevel=2,
            )
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
        if self.name_pattern != _DEFAULT_NAME_PATTERN:
            try:
                return re.findall(self.name_pattern, code, re.IGNORECASE)[0]
            except Exception:
                return ""
        try:
            return _main_class_name(code)
        except Exception:
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

        pattern = self.cs_pattern
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
        if self.code_pattern != _DEFAULT_CODE_PATTERN:
            match = re.search(self.code_pattern, message, re.DOTALL | re.IGNORECASE)
            code = match.group(1) if match else None
        else:
            code = _extract_python_block(message)
        if code is not None:
            return code
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
        match = re.search(self.desc_pattern, message, re.IGNORECASE)
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

    def __init__(
        self,
        api_key=None,
        model="gpt-4-turbo",
        temperature=0.8,
        base_url: str | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float | None = None,
        **kwargs,
    ):
        """
        Initializes the LLM manager with an API key and model name.

        Also works with OpenAI-compatible servers such as vLLM, SGLang or
        llama.cpp: set ``base_url`` to the server (for example
        ``http://localhost:8000/v1``) and ``model`` to the served model name.

        Args:
            api_key (str | None): API key. ``None`` reads ``OPENAI_API_KEY``;
                with a ``base_url`` and no key, ``"EMPTY"`` is sent (local
                servers usually accept any key).
            model (str, optional): model name. Defaults to "gpt-4-turbo".
            temperature (float, optional): sampling temperature.
            base_url (str, optional): URL of an OpenAI-compatible API.
            extra_body (dict, optional): extra fields sent with every
                request, for server-specific options; for example vLLM's
                ``{"chat_template_kwargs": {"enable_thinking": False}}`` or
                ``{"max_tokens": 4096}``.
            timeout (float, optional): seconds before a request times out
                (the client's default if None).
        """
        if api_key is None and base_url:
            api_key = "EMPTY"
        super().__init__(api_key, model, base_url, **kwargs)
        self._client_kwargs = dict(api_key=api_key)
        if base_url:
            self._client_kwargs["base_url"] = base_url
        if timeout is not None:
            self._client_kwargs["timeout"] = timeout
        self.extra_body = dict(extra_body) if extra_body else None
        self.client = openai.OpenAI(**self._client_kwargs)
        logging.getLogger("openai").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("httpx2").setLevel(logging.ERROR)  # openai >= 3
        self.temperature = temperature

    def get_config(self) -> list[dict[str, Any]]:
        llm_config = self._client_kwargs.copy()
        llm_config["temperature"] = self.temperature
        llm_config.pop("api_key")
        if self.extra_body:
            llm_config["extra_body"] = self.extra_body
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

        options = dict(kwargs)  # kept intact for every retry
        temperature = options.pop("temperature", self.temperature)
        extra_body = getattr(self, "extra_body", None)
        if extra_body and "extra_body" not in options:
            options["extra_body"] = extra_body

        attempt = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=session_messages,
                    temperature=temperature,
                    **options,
                )
                return response.choices[0].message.content

            except openai.RateLimitError as err:
                attempt += 1
                if attempt > max_retries:
                    raise
                time.sleep(_retry_after_seconds(err, default_delay * attempt))

            except (
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.APIError,
            ) as err:
                # Bad requests, authentication, unknown models, ...: retrying
                # cannot help.
                if not _retryable_status(getattr(err, "status_code", None)):
                    raise
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
        kwargs.pop("base_url", None)
        super().__init__(
            api_key,
            model=model,
            temperature=temperature,
            base_url="https://api.deepseek.com",
            **kwargs,
        )


def _gemini_retryable(err) -> bool:
    """Whether a google-genai error is worth retrying."""
    code = getattr(err, "code", None)
    if isinstance(code, int):
        return _retryable_status(code)
    # Older google-api-core errors (ResourceExhausted) carry a retry delay.
    if getattr(err, "retry_delay", None) is not None:
        return True
    if re.search(r"\b429\b|ResourceExhausted|RESOURCE_EXHAUSTED", str(err)):
        return True
    if isinstance(err, (ConnectionError, TimeoutError)):
        return True
    # Network errors: google-genai >= 3 uses httpx2, older versions httpx.
    for module in ("httpx2", "httpx"):
        try:
            transport_error = importlib.import_module(module).TransportError
        except (ImportError, AttributeError):  # pragma: no cover
            continue
        if isinstance(err, transport_error):
            return True
    return False


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
        # google-genai expects roles "user"/"model" and parts as {"text": ...}.
        history = [
            {
                "role": "model" if m["role"] in ("assistant", "model") else "user",
                "parts": [{"text": m["content"]}],
            }
            for m in session_messages[:-1]
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
                if not _gemini_retryable(err):
                    raise  # invalid request, key or model: retrying cannot help
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
                time.sleep(_retry_after_seconds(err, default_delay * attempt))

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
        # lms is a placeholder when lmstudio is not installed.
        self.llm = lms.llm(self.model) if hasattr(lms, "llm") else None

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
        state = self.__dict__.copy()  # do not strip the live object
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
