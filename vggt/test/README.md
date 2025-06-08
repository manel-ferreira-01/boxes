## Use script test_simplebox_vggt.ipynb to test the component

verify the port number and the IP first

### Building the message (see the notebook in the test folder)

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
