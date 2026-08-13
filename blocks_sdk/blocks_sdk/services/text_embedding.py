from blocks_sdk.client import Client


class TextEmbedding(Client):
    """Text embedding service (SBERT)."""
    
    def __init__(self, address="text_embed:8061", **kwargs):
        super().__init__(address, **kwargs)
    
    def encode(self, texts, **kwargs):
        """Encode text to embeddings."""
        if not isinstance(texts, list):
            texts = [texts]
        
        return self.call(
            method="Forward",
            data={"sentences": texts},
            **kwargs
        )
