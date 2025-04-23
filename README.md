# Generic processing component grpc enabled
This code builds a AI4EU (acumos) component that executes the processing defined in a specific python function.

The input and return value are always a **.mat** file binary coded. 

Data to the service is passed and returned through variables stored inside the .mat files (loaded and saved with **scipy.io.loadmat/savemat** )
## The specific code of the component
Edit file src/simplebox_service.py and place your edits in function run_codigo

```python 
def run_codigo(datafile):
    """
    Reads all variables from a MATLAB .mat file,

    Parameters:
    datafile (bytes pointer): Data sent through grpc (matfile message - data field)

    Returns:
    new_file_pointer (io.BytesIO): In-memory file-like object containing the output in a .mat file.
    """
    #Load the mat file using scipy.io.loadmat
    mat_data=loadmat(io.BytesIO(datafile))
    
    # SPECIFIC CODE STARTS HERE - 
    # exemple (assume there is a variable im in the mat file
    # image=mat_data["im"]



    # SPECIFIC CODE ENDS HERE

    # create file to return data
    f=io.BytesIO()

    # WRITE RETURNING DATA TO MAT FILE - be carefull with the variable naming
    #example (create a variable im ):  savemat(f,{"im":-image})
    return f.getvalue()
```

## Creating a component
Pull the docker image :
```shell
docker pull jpcosteira/genericbox
```
or run the script  
```shell
#you may edit the script and rename de image
bash buildme
```

## Launching the service
* **Standalone:** 
  1. assign a port number from the host that maps to port 8061 exposed by the docker container (mandatory for AI4EU pipelines).
  1. Launch the component: 
  1. ```shell
     $ docker run --rm -it -p 8061:8061 --name servicename -v pathtoexternal/externalfile.py=/workspace/external.py jpcosteira/genericbox:<specific tag>-latest```
 1. Use a grpc enabled code to test and/or interact with the service. See notebook ```test/test_image_generic.ipynb```

* **Running the service in a pipeline:** Follow the configuration and deployment rules of ```maestro``` the pipeline orchestrator [maestro@github](https://github.com/jpcosteira/maestro)

