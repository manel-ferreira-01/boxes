## Download the vggt directory 
https://github.com/facebookresearch/vggt.git

## Code inside the box 

File simplebox_service.py
### In the service init

download the model and copy to GPU

```python
class ServiceImpl(simplebox_pb2_grpc.SimpleBoxServiceServicer):

    def __init__(self):
        """
        Args:
            
          Loads VGGT model 
        """
        

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

    # Initialize the model and load the pretrained weights.
    # This will automatically download the model weights the first time it's run, which may take a while.
        self._model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)        
``` 

### In the method process of the service (see protobuf)

calls run_codigo with arguments to run vggt

```python
    def process(self, request: simplebox_pb2.matfile, context):
        """
        matfile is the matlab file with input data

        Args:
            request: The ImageAndFeatures request to process
            context: Context of the gRPC call

        Returns:
            The Image with the applied function
        features={'kp','desc'}
        """
        datain = request.data

        ret_file= run_codigo(datain, self._model,self._device)
        return simplebox_pb2.matfile(data=ret_file)
```

### In function run_codigo

Reads the image data and processes the sequence. Note that inside the mat file there must be a **imgdata** list with the images.

```python
def run_codigo(datafile,model,device):
  .....
  mat_contents = loadmat('datafile')

  # Extract and flatten the imgdata array
  imgdata = mat_contents['imgdata'].squeeze()
```

