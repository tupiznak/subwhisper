from typing import Literal, Any

from transformers import pipeline

whisper_types = Literal['tiny', 'small', 'base', 'medium', 'large-v2']


def whisper_pipeline(*, whisper: whisper_types = 'small', device: int = 0, batch_size: int = 4) -> pipeline:
    transcribe = pipeline(
        task="automatic-speech-recognition",
        model=f"openai/whisper-{whisper}",
        chunk_length_s=30,
        device=device,
        batch_size=batch_size
    )
    transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(
        language="russian",
        task="transcribe"
    )
    return transcribe


def setup(*, whisper: whisper_types = 'small', device: int = 0, batch_size: int = 4) -> None:
    whisper_pipeline(whisper=whisper, device=device, batch_size=batch_size)


def process(*, file: str = None, whisper: whisper_types = 'small', device: int = 0, batch_size: int = 4) \
        -> dict[str, list[dict[str, Any]]]:
    pipe = whisper_pipeline(whisper=whisper, device=device, batch_size=batch_size)
    transcribed_chunks = pipe(file, return_timestamps="word")['chunks']

    events = [
        {'tStartMs': int(chunk['timestamp'][0] * 1000),
         'segs': [{'utf8': chunk['text']}]}
        for chunk in transcribed_chunks
    ]
    json3 = {'events': events}
    return json3


if __name__ == '__main__':
    process()
