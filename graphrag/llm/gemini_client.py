import json
import time
from typing import List, Dict, Any, Type, TypeVar, Optional

# Nuevas importaciones del SDK 2.0
from google import genai
from google.genai import types
from pydantic import BaseModel
from ..config import get_gemini_settings

T = TypeVar('T', bound=BaseModel)

class GeminiClient:
    def __init__(self):
        self.settings = get_gemini_settings()
        # Instanciamos el cliente en lugar de configurar globalmente
        self.client = genai.Client(api_key=self.settings.gemini_api_key, http_options={'timeout': 60_000})

    def _parse_messages(self, messages: List[Dict[str, str]]) -> tuple[Optional[str], List[types.Content]]:
        """Adapta el historial al formato estricto de types.Content de la nueva API."""
        system_instruction = None
        history = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                system_instruction = content
            else:
                gemini_role = "model" if role == "assistant" else "user"
                # Creamos el objeto Content nativo de la nueva librería
                history.append(
                    types.Content(
                        role=gemini_role, 
                        parts=[types.Part.from_text(text=content)]
                    )
                )

        return system_instruction, history

    def _rate_limit_delay(self):
        """
        Pausa de 5 segundos antes de cada llamada.
        """
        time.sleep(1)

    def chat(
            self,
            messages: List[Dict[str, str]],
            model: str = None,
            temperature: float = 0.0,
            format: str = None
    ) -> str:
        """Genera una respuesta de texto estándar o JSON libre."""
        model_name = model or self.settings.gemini_model
        system_instruction, history = self._parse_messages(messages)

        config_kwargs = {"temperature": temperature}

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        if format == "json":
            config_kwargs["response_mime_type"] = "application/json"

        # Aplicamos la configuración en el nuevo formato
        config = types.GenerateContentConfig(**config_kwargs)
        
        self._rate_limit_delay()

        response = self.client.models.generate_content(
            model=model_name,
            contents=history,
            config=config
        )
        return response.text

    def structured_output(
            self,
            prompt: str,
            schema: Type[T],
            system_prompt: str = "You are an expert data extractor. Respond strictly in JSON format.",
            model: str = None
    ) -> T:
        """Fuerza al modelo a devolver un objeto basado en un esquema de Pydantic."""
        model_name = model or self.settings.gemini_model
        
        config = types.GenerateContentConfig(
            temperature=0.0,
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=schema,
            max_output_tokens=2000
        )

        self._rate_limit_delay()

        response = self.client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config
        )

        return schema.model_validate_json(response.text)

    def structured_output_with_chat(
            self,
            messages: List[Dict[str, str]],
            schema: Type[T],
            model: str = None,
            temperature: float = 0.0
    ) -> T:
        """Genera output estructurado sobre un esquema de Pydantic, manteniendo el contexto."""
        model_name = model or self.settings.gemini_model
        system_instruction, history = self._parse_messages(messages)

        config_kwargs = {
            "temperature": temperature,
            "response_mime_type": "application/json",
            "response_schema": schema
        }

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        config = types.GenerateContentConfig(**config_kwargs)

        self._rate_limit_delay()

        response = self.client.models.generate_content(
            model=model_name,
            contents=history,
            config=config
        )

        return schema.model_validate_json(response.text)

    def embed(self, texts: List[str], model: str = "text-embedding-004") -> List[List[float]]:
        """Genera embeddings usando la nueva API de cliente."""
        self._rate_limit_delay()
        
        response = self.client.models.embed_content(
            model=model,
            contents=texts
        )
        
        # La nueva API devuelve una lista de objetos Embedding en response.embeddings
        return [emb.values for emb in response.embeddings]

    @staticmethod
    def extract_json(response: str) -> Dict[str, Any]:
        """Extrae JSON de la respuesta."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            start = response.find('```json')
            if start != -1:
                start = response.find('\n', start) + 1
                end = response.find('```', start)
                if end != -1:
                    return json.loads(response[start:end].strip())

            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1:
                return json.loads(response[start:end + 1])

            raise ValueError("No se pudo extraer JSON de la respuesta")