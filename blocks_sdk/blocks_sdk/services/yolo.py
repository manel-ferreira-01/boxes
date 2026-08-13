from blocks_sdk.client import Client


class YOLO(Client):
    """YOLO detection and tracking service."""
    
    def __init__(self, address="yologrpc:8061", **kwargs):
        super().__init__(address, **kwargs)
    
    def detect(self, images, threshold=0.5, **kwargs):
        """Run YOLO detection on a batch of images."""
        if not isinstance(images, list):
            images = [images]
        
        return self.call(
            method="DetectSequence",
            data={"images": images},
            config_json=f'{{"threshold": {threshold}}}',
            **kwargs
        )
    
    def track(self, frames, stream_id=0, **kwargs):
        """Run YOLO tracking on a sequence of images."""
        if not isinstance(frames, list):
            frames = [frames]
        
        return self.call(
            method="TrackSequence",
            data={"images": frames},
            config_json=f'{{"stream": {stream_id}}}',
            **kwargs
        )
