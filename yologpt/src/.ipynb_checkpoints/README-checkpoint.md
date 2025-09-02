 ## Details about the services and methods

 TODO: establish a set of arguments that fully customize the call to yolo

 - model
 - arguments: reset id, confidence threshold ...

```python
            # Parse track config JSON
            try:
                config = json.loads(request.track_config_json or "{}")
                if config:
                    configyolo=config[1]
                    if config[0].pop("reset",None):#process commands (TODO)
                        self.model.predictor.trackers[0].reset() #reset tracker ID's
                else :
                    configyolo={}

            except json.JSONDecodeError:
                config = {}
                logging.exception("JSON not valid ")
               # Run tracking
            results = self.model.track(source=frame, persist=True, **configyolo)
````

## Format of the JSON 

Return message from yolo is a list of dicts with main keys (aispgradio and YOLO). In the future the two should be separated and merged in the end only.

```python
msg=[dict("aispgradio"),dict("YOLO")])
````
### Track

configtrack=l['aispgradio']['yolotrack']['reset'] to start new features (ID) or continue previous sequence

case: 

```python
[
    {
        "aispgradio": {
            "command": "single",
            "user": "",
            "input_count": 2,
            "timestamp": "2025-08-28T08:14:34.039162",
            "yoloconfig": "vai aqui a configura\u00e7ao"
        }
    },
    {
        "YOLO": [
            [
                {
                    "bbox": [
                        329.5801696777344,
                        45.80845642089844,
                        1116.6998291015625,
                        952.8724975585938
                    ],
                    "confidence": 0.882569432258606,
                    "class_id": 0
                },
                {
                    "bbox": [
                        0.9119682312011719,
                        549.10302734375,
                        265.2938232421875,
                        953.0321044921875
                    ],
                    "confidence": 0.8661189675331116,
                    "class_id": 56
                },
                {
                    "bbox": [
                        376.55194091796875,
                        354.4136962890625,
                        1083.3741455078125,
                        955.0074462890625
                    ],
                    "confidence": 0.7086978554725647,
                    "class_id": 0
                },
                {
                    "bbox": [
                        6.99664306640625,
                        0.6308555603027344,
                        1424.30908203125,
                        143.38804626464844
                    ],
                    "confidence": 0.3194182813167572,
                    "class_id": 25
                }
            ]
        ]
    }
]
````