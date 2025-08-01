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

   