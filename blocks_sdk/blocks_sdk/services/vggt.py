from blocks_sdk.client import Client


class VGGT(Client):
    """VGGT 3D reconstruction service."""
    
    def __init__(self, address="vggtgrpc:8061", **kwargs):
        super().__init__(address, **kwargs)
    
    def reconstruct(self, images, **kwargs):
        """Run 3D reconstruction from multiple images."""
        if not isinstance(images, list):
            images = [images]
        
        return self.call(
            method="Process",
            data={"images": images},
            config_json='{"task": "reconstruction"}',
            **kwargs
        )
