import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from service import main, prompt


@pytest.fixture(autouse=True)
def reset_tokenizer(monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.delenv("LENGTH_TOKENIZER_PATH", raising=False)
    monkeypatch.setattr(prompt, "_LENGTH_TOKENIZER", None)
    monkeypatch.setattr(prompt, "_LENGTH_TOKENIZER_PATH", None)


def model_dir(tmp_path):
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    return tmp_path


def test_reads_tokenizer_from_model_path_and_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_dir(tmp_path)))
    assert prompt.model_token_length("hello") == 1
    assert prompt.model_token_length("hello hello") == 2
    loaded = prompt._load_length_tokenizer()
    assert prompt._load_length_tokenizer() is loaded
    assert prompt.length_tokenizer_path() == tmp_path / "tokenizer.json"


def test_explicit_file_overrides_model_path(tmp_path, monkeypatch):
    model_dir(tmp_path)
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "unavailable-gpu-host-model"))
    monkeypatch.setenv("LENGTH_TOKENIZER_PATH", str(tmp_path / "tokenizer.json"))
    assert prompt.model_token_length("hello") == 1


def test_missing_configuration_is_explicit():
    with pytest.raises(ValueError, match="MODEL_PATH"):
        prompt.model_token_length("hello")


def test_missing_file_does_not_use_character_estimate(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        prompt.model_token_length("hello")


@pytest.mark.asyncio
async def test_startup_validates_model_tokenizer(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        async with main.lifespan(main.app):
            pytest.fail("Startup must not complete without the tokenizer")
    model_dir(tmp_path)
    async with main.lifespan(main.app):
        assert prompt.model_token_length("hello") == 1
