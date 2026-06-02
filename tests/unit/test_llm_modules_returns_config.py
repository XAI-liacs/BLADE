from iohblade.llm import Ollama_LLM, Gemini_LLM, OpenAI_LLM
from unittest.mock import MagicMock, patch
def test_ollama_returns_config():
    llm = Ollama_LLM(
        'project:eve'
    )
    configs = llm.get_config()
    for config in configs:
        assert 'model' in config
        assert 'config' in config
        assert 'hardware' in config
        assert config['model'] == 'project:eve'


def test_gemini_returns_config():
    llm = Gemini_LLM(
        '1234567890',
        model='gemini-4.0-flash'
    )
    for config in llm.get_config():
        assert 'model' in config
        assert 'config' in config
        assert 'hardware' in config
        assert config['hardware'] == {}     #Server provided llm, we don't know where it is from.
        assert config['model'] == 'gemini-4.0-flash'
        assert 'api_key' not in config['config']

def test_openai_returns_config():
    llm = OpenAI_LLM(
        'gpt-4-turbo'
    )
    for config in llm.get_config():
        assert 'model' in config
        assert 'config' in config
        assert 'hardware' in config
        assert config['hardware'] == {}     #Server provided llm, we don't know where it is from.
        assert config['model'] == 'gpt-4-turbo'
        assert 'api_key' not in config['config']

try:
    from iohblade.llm import MLX_LM_LLM
except:
    MLX_LM_LLM = None

def test_mlx_lm_returns_config():
    if MLX_LM_LLM is None:
        return
    with patch('iohblade.llm.load') as mock_llm:
        mock_instance = MagicMock()
        mock_llm.return_value = (mock_instance, mock_instance)
        llm = MLX_LM_LLM('mlx_optimized')
        for config in llm.get_config():
            assert 'model' in config
            assert 'config' in config
            assert 'hardware' in config
            assert config['model'] == 'mlx_optimized'
            assert 'api_key' not in config['config']

try: 
    from iohblade.llm import LMStudio_LLM
except:
    LMStudio_LLM = None

def test_lmstudio_returns_config():
    if LMStudio_LLM is None:
        return
    with patch("iohblade.llm.lms.llm") as mock_llm:
        mock_instance = MagicMock()
        mock_llm.return_value = mock_instance
        llm = LMStudio_LLM("mlx_lm:optimised")

        mock_instance.some_method.return_value = "dummy"

        for config in llm.get_config():
            assert "model" in config
            assert "config" in config
            assert "hardware" in config
            assert config['model'] == 'mlx_lm:optimised'
            assert 'api_key' not in config['config']

