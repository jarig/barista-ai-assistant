import logging
from io import BytesIO
from typing import List, IO, Optional, Literal, Union
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionMessageParam, \
    ChatCompletionSystemMessageParam, ChatCompletionAssistantMessageParam, ChatCompletionToolMessageParam, \
    ChatCompletion

from src.aiclient.models import AssistantHistoryMessage, AssistantResult

logger = logging.getLogger("vita.aiclient")


class OpenAiClient:
    class Models:
        TEXT_TO_SPEECH = "tts-1"
        SPEECH_TO_TEXT = "whisper-1"
        CHAT_MODEL = "gpt-4"

    def __init__(self, key: str, user: str = "User", initial_prompt: str = None):
        self._client = OpenAI(api_key=key)
        self._history: List[AssistantHistoryMessage] = []
        self._user = user
        self._inital_prompt = initial_prompt
        if initial_prompt:
            self.add_message(AssistantHistoryMessage(role=AssistantHistoryMessage.Role.System, message=initial_prompt))

    def add_message(self, msg: AssistantHistoryMessage):
        self._history.append(msg)

    def get_messages(self, ignore_roles = None):
        if ignore_roles is None:
            ignore_roles = []
        return filter(lambda m: m.role not in ignore_roles, self._history)

    def convert_to_speech(self, text, response_fmt: Literal["mp3", "opus", "aac", "flac"] = "mp3",
                          voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = "alloy",
                          speed: float = 1.0) -> BytesIO:
        response = self._client.audio.speech.create(
            model=OpenAiClient.Models.TEXT_TO_SPEECH,
            voice=voice,
            response_format=response_fmt,
            input=text,
            speed=speed,
        )
        chunk_size = 512
        result = BytesIO()
        for data in response.iter_bytes(chunk_size):
            result.write(data)
        result.seek(0)
        return result

    def speech_to_text(self, audio_bytes: Union[IO[bytes], BytesIO], prompt: Optional[str] = None) -> str:
        kwargs = {}
        if prompt is None:
            prompt = self._inital_prompt
        if prompt:
            kwargs["prompt"] = prompt
        if not hasattr(audio_bytes, "name"):
            # random name, just to tell of what type data is
            audio_bytes.name = "request.wav"
        transcript = self._client.audio.transcriptions.create(
            model=OpenAiClient.Models.SPEECH_TO_TEXT,
            file=audio_bytes,
            **kwargs
        )
        return transcript.text

    def text_prompt(self, prompt: str, temperature: float = 0.3,
                    max_tokens: int = 1500) -> Optional[AssistantResult]:
        _retries = 3
        self._history.append(AssistantHistoryMessage(role=AssistantHistoryMessage.Role.User, message=prompt))
        response: Optional[ChatCompletion] = None
        while _retries > 0:
            try:
                response = self._client.chat.completions.create(user=self._user,
                                                                model=OpenAiClient.Models.CHAT_MODEL,
                                                                temperature=temperature, max_tokens=max_tokens,
                                                                messages=self._get_messages_list(self._history))
                break
            except Exception as ex:
                logger.exception(f"Assistant failed to answer the ask: {prompt}: {ex}")
                _retries -= 1
                if not _retries:
                    # no more retries left
                    break

        result = None
        if response and response.choices:
            state = AssistantResult.State.Stop
            finish_reason = response.choices[0].finish_reason
            if finish_reason == AssistantResult.State.Length:
                state = AssistantResult.State.Length
            elif finish_reason == AssistantResult.State.Null:
                state = AssistantResult.State.Null
            result = AssistantResult(response.choices[0].message.content, state=state)

        logger.info("assistant_ask", extra={
            'custom_dimensions': {
                'retriesLeft': _retries,
                'hasResponse': result is not None,
                'state': result.state if result else None
            }
        })
        return result

    @staticmethod
    def _get_messages_list(history: List[AssistantHistoryMessage]):
        result: List[ChatCompletionMessageParam] = []
        for m in history:
            if m.role == AssistantHistoryMessage.Role.User:
                msg_obj = ChatCompletionUserMessageParam(content=m.message, role="user")
            elif m.role == AssistantHistoryMessage.Role.System:
                msg_obj = ChatCompletionSystemMessageParam(content=m.message, role="system")
            elif m.role == AssistantHistoryMessage.Role.Assistant:
                msg_obj = ChatCompletionAssistantMessageParam(content=m.message, role="assistant")
            else:
                msg_obj = ChatCompletionToolMessageParam(content=m.message, role="tool", tool_call_id="ai_client")
            result.append(msg_obj)
        return result
