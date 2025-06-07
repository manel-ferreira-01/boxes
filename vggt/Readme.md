# 3D reconstruction with VGGT:
### Building the container:
Check docker directory. Add the requirements listed in 
https://github.com/facebookresearch/vggt.git

### Data format 


- **Input** : a mat file with a dictionary with a list of encoded images
- **Output** : a mat file with all data returned by vggt

### Building the message (see test notebook)

file_paths is a list with the image files to be sent to the "box" and images should be stored in a list named **imgdata** .
```python
file_paths = [
    'path/to/image1.jpg',
    'path/to/image2.jpg',
    'path/to/image3.jpg'
]

# List to hold binary data
imgdata = []

# Read binary contents of each file and append to imgdata
for path in file_paths:
    with open(path, 'rb') as f:
        data = f.read()
        imgdata.append(bytearray(data))  # Use bytearray for .mat compatibility

# Save the imgdata list to a .mat file
scipy.io.savemat('imgdata.mat', {'imgdata': imgdata})
```

## Code inside the box 
Reads the image data and processes the sequence. Note that inside the mat file there must be a **imgdata** list with the images.

```python
mat_contents = loadmat('datafile')

# Extract and flatten the imgdata array
imgdata = mat_contents['imgdata'].squeeze()
```
