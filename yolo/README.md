# Yolo generic component grpc enabled
This code builds a AI4EU (acumos) Yolo component that executes the processing defined in the "simplebox" calling function. This component is an adaptation of the simplebox http://github.com/jpcosteira/simplebox.git

The input and return value are always a **.mat** file. 

Data to the service is passed and returned through variables stored inside the .mat files (loaded and saved with **scipy.io.loadmat/savemat** )
## The specific code of YOLO. 
Note that inside the mat file there should be a variable named im (image).
Returns the "results" object flatened to a single entry dictionary
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


    im=mat_data["im"]
    results=model(im)

    all_arrays = []

    for i, result in enumerate(results):
        arr_dict = extract_array_attributes(result, prefix=f'result[{i}].')
        all_arrays.append(arr_dict)


    # SPECIFIC CODE ENDS HERE

    f=io.BytesIO()
    # WRITE RETURNING DATA
    savemat(f,{"results":all_arrays})
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

