from embed_providers import OpenAIEmbedder

def get_embedder(provider: str, **kwargs):
    if provider == "openai":
        return OpenAIEmbedder(api_key=kwargs["api_key"], model=kwargs["model"])
    else:
        raise ValueError(f"Unknown provider: {provider}")