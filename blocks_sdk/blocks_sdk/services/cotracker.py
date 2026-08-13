from blocks_sdk.client import Client


class CoTracker(Client):
    """CoTracker video motion tracking service."""
    
    def __init__(self, address="cotracker:8061", **kwargs):
        super().__init__(address, **kwargs)
    
    def track_video(self, video=None, video_frames=None, **kwargs):
        """Track motion in video frames."""
        v = video if video is not None else video_frames
        if isinstance(v, list) and v and isinstance(v[0], bytes):
            v = b"".join(v)
            
        data = {}
        if v is not None:
            data["video"] = v
            
        return self.call(
            method="Forward",
            data=data,
            **kwargs
        )
