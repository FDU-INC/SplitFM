from abc import ABC, abstractmethod
from typing import Iterable

class SplitModelAdapter(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def load(self, weights_dir: str):
        """
        Load model configuration and tokenizer
        Initialize and load pre-trained models
        Move models to GPU and convert to half precision
        :param weights_dir: Storage directory of weight files
        """
        pass

    @abstractmethod
    def infer(self, input_sentence: str, **kwargs) -> str:
        """
        inference
        :param input_sentence: The input text
        :param kwargs: Other optinonal arguments
        :return: The generated text result
        """
        pass

    def stream_infer(self, input_sentence: str, **kwargs) -> Iterable[str]:
        """
        Incrementally yield generated text pieces for streaming responses.
        Subclasses can override this without changing the existing infer() path.
        """
        raise NotImplementedError("This adapter does not implement streaming inference.")
