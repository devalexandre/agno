from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agno.embedder.base import Embedder
from agno.utils.log import logger

try:
    from ollama import Client as OllamaClient
    import importlib.metadata as metadata
    from packaging import version

    ollama_version = metadata.version("ollama")
    if version.parse(ollama_version).major == 0 and version.parse(ollama_version).minor < 3:
        import warnings
        warnings.warn("Suporte apenas para Ollama v0.3.x ou superior", UserWarning)
        raise RuntimeError("Versão incompatível do Ollama detectada.")

except ImportError as e:
    if "ollama" in str(e):
        raise ImportError(
            "Ollama não instalado. Instale com `pip install ollama`"
        ) from e
    else:
        raise ImportError(
            "Dependências faltantes. Instale com `pip install packaging importlib-metadata`"
        ) from e
        
@dataclass
class OllamaEmbedder(Embedder):
    id: str = "openhermes"
    dimensions: int = 4096
    host: Optional[str] = None
    timeout: Optional[Any] = None
    options: Optional[Any] = None
    client_kwargs: Optional[Dict[str, Any]] = None
    ollama_client: Optional[OllamaClient] = None

    @property
    def client(self) -> OllamaClient:
        if self.ollama_client:
            return self.ollama_client

        _ollama_params: Dict[str, Any] = {}
        if self.host:
            _ollama_params["host"] = self.host
        if self.timeout:
            _ollama_params["timeout"] = self.timeout
        if self.client_kwargs:
            _ollama_params.update(self.client_kwargs)
        return OllamaClient(**_ollama_params)

    def _response(self, text: str) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if self.options is not None:
            kwargs["options"] = self.options

        return self.client.embeddings(prompt=text, model=self.id, **kwargs)  # type: ignore

    def get_embedding(self, text: str) -> List[float]:
        try:
            response = self._response(text=text)
            if response is None:
                return []
            return response.get("embedding", [])
        except Exception as e:
            logger.warning(e)
            return []

    def get_embedding_and_usage(self, text: str) -> Tuple[List[float], Optional[Dict]]:
        embedding = self.get_embedding(text=text)
        usage = None
        return embedding, usage
