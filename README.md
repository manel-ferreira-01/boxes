# Generic processing component grpc enabled
This code builds a AI4EU (acumos) component that executes the processing defined by an external funcition.

This component is a modification of the flexible opencv component (see https://github.com/DuarteMRAlves/opencv-grpc-service).
It accepts as input a **.mat** file and returs another **.mat** file both binary coded. 

Data to the service is passed and returned through variables stored inside the .mat files (loaded and saved with **scipy.io.loadmat/savemat** )
f=io.BytesIO()
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
## Processing
Edit a .py file (ex. externalfile.py) and insert the code you want to execute.

Main steps:
- ```import``` numpy, scipy, io and whatever other modules/libraries you need
- There must be a function named as ```def calling_function(datafile):```
- Load the .mat file ```mat_data=loadmat(io.BytesIO(datafile))```
- **Read the variables of interest from the dictionary ```mat_data``` and do all the processing your service must do**
- Output: 1) Create the output file 2) Save your results in a .mat file and 3) Return the byte content of the file
  ```python
     retfile=io.BytesIO()
     savemat(retfile,{"var1":var1,"var2":var2,...,"varn":varn})
     return retfile.getvalue()
  ```
- **Or do it *the easy way*:** copy ```src/external1.py``` and edit between the two Comments # SPECIFIC CODE STARTS HERE and # SPECIFIC CODE ENDS HERE 

## Launching the service
* **Standalone:** 
  1. assign a port number from the host that maps to port 8061 exposed by the docker container (mandatory for AI4EU pipelines).
  1. Launch the component: 
  1. ```shell
     $ docker run --rm -it -p 8061:8061 --name servicename -v pathtoexternal/externalfile.py=/workspace/external.py jpcosteira/genericbox:<specific tag>-latest```
 1. Use a grpc enabled code to test and/or interact with the service. See notebook ```test/test_image_generic.ipynb```

* **Running the service in a pipeline:** Follow the configuration and deployment rules of ```maestro``` the pipeline orchestrator [maestro@github](https://github.com/jpcosteira/maestro)

## Todo
Change all names of files and variables from image_generic or Image to something more generic !
