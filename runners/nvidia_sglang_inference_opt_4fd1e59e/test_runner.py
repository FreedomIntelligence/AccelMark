from __future__ import annotations

from runners.benchmark_runner import InferenceRequest

from runner import SGLangInferenceOptimizedRunner, SGLangRunner


class _Tokenizer:
    chat_template = "present"

    def __init__(self):
        self.encode_calls = 0
        self.batch_encode_calls = 0

    def apply_chat_template(self, conversations, **_kwargs):
        return [f"<user>{conversation[0]['content']}<assistant>" for conversation in conversations]

    def encode(self, text):
        self.encode_calls += 1
        return text.split()

    def __call__(self, texts, *, add_special_tokens):
        assert add_special_tokens is True
        self.batch_encode_calls += 1
        return {"input_ids": [text.split() for text in texts]}


class _Engine:
    def __init__(self, outputs):
        self.outputs = outputs
        self.prompts = None

    def generate(self, *, prompt, sampling_params):
        self.prompts = prompt
        assert sampling_params == {"temperature": 0.0}
        return self.outputs


def _request(index: int, prompt: str) -> InferenceRequest:
    return InferenceRequest(
        prompt=prompt,
        request_id=index,
        input_tokens=index + 1,
    )


def test_offline_fast_path_batches_formatting_and_uses_metadata():
    runner = SGLangInferenceOptimizedRunner()
    runner.tokenizer = _Tokenizer()
    runner._sampling_params = {"temperature": 0.0}
    runner.engine = _Engine(
        [
            {"text": "alpha beta", "meta_info": {"completion_tokens": 2}},
            {"text": "gamma", "meta_info": {"completion_tokens": 1}},
        ]
    )

    results = runner.inference_fn_offline(
        [_request(0, "first"), _request(1, "second")]
    )

    assert runner.engine.prompts == [
        "<user>first<assistant>",
        "<user>second<assistant>",
    ]
    assert [result.output_tokens for result in results] == [2, 1]
    assert [result.input_tokens for result in results] == [1, 2]
    assert [result.output_text for result in results] == ["alpha beta", "gamma"]
    assert runner.tokenizer.encode_calls == 0
    assert runner.tokenizer.batch_encode_calls == 1

    runner.inference_fn_offline(
        [_request(0, "first"), _request(1, "second")]
    )
    assert runner.tokenizer.batch_encode_calls == 1


def test_offline_fast_path_falls_back_to_encoding_old_outputs():
    runner = SGLangInferenceOptimizedRunner()
    runner.tokenizer = _Tokenizer()
    runner._sampling_params = {"temperature": 0.0}
    runner.engine = _Engine([{"text": "one two three"}])

    result = runner.inference_fn_offline([_request(0, "first")])[0]

    assert result.output_tokens == 3
    assert runner.tokenizer.encode_calls == 0
    assert runner.tokenizer.batch_encode_calls == 1


def test_load_policy_rejects_invalid_continuous_steps():
    runner = SGLangInferenceOptimizedRunner()
    runner._current_scenario = "offline"
    runner._runner_config = {"continuous_decode_steps": 0}

    try:
        runner.load_model("unused", {})
    except ValueError as error:
        assert "continuous_decode_steps" in str(error)
    else:
        raise AssertionError("invalid continuous_decode_steps was accepted")


def test_ngram_branch_is_strictly_larger_than_window(monkeypatch):
    captured = {}

    def capture_load_model(self, _model_path, _parallelism):
        captured.update(self._runner_config["engine_kwargs"])

    monkeypatch.setattr(SGLangRunner, "load_model", capture_load_model)
    runner = SGLangInferenceOptimizedRunner()
    runner._current_scenario = "offline"
    runner._runner_config = {
        "speculative_mode": "ngram",
        "ngram_window": 4,
        "ngram_breadth": 1,
    }

    runner.load_model("unused", {})

    assert captured["speculative_ngram_max_match_window_size"] == 4
    assert captured["speculative_ngram_branch_length"] == 5
