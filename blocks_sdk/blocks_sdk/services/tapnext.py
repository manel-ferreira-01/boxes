import json
from blocks_sdk.client import Client


class TAPNext(Client):
    """TAPNext video tracking service."""
    
    def __init__(self, address="tapnext:8061", **kwargs):
        super().__init__(address, **kwargs)
    
    def track(self, video_frames, grid_size=32, **kwargs):
        """Track points in video frames using TAPNext."""
        if not isinstance(video_frames, list):
            video_frames = [video_frames]
        
        config = json.dumps({
            "tapnext": {
                "parameters": {"grid_size": grid_size}
            }
        })
        
        return self.call(
            method="Process",
            data={"images": video_frames},
            config_json=config,
            **kwargs
        )
    
    def reset(self, **kwargs):
        """Reset tracking state."""
        return self.call(
            method="Process",
            data={},
            config_json='{"tapnext": {"command": "reset"}}',
            **kwargs
        )
