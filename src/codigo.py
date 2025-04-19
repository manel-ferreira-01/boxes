import io
from scipy.io import loadmat, savemat
import numpy as np
import cv2


def codigo(datafile):
    """
    Reads all variables from a MATLAB .mat file given a file pointer,
    
    Parameters:
    file_pointer (file-like object): Opened .mat file in binary read mode.

    Returns:
    new_file_pointer (io.BytesIO): In-memory file-like object containing the cloned .mat file.
    """
    #Load the mat file using scipy.io.loadmat
    mat_data=loadmat(io.BytesIO(datafile))

    # SPECIFIC CODE STARTS HERE
    im2=mat_data["im"]




    # SPECIFIC CODE ENDS HERE

    f=io.BytesIO()
    # WRITE RETURNING DATA
    savemat(f,{"im":im2,"kp":keypoints_array,"desc":descriptor})
    return f.getvalue()

# Load all data from the input .mat file
    data = scipy.io.loadmat(file_pointer)

    # Remove metadata keys (which start with __) commonly found in .mat files
    clean_data = {k: v for k, v in data.items() if not k.startswith('__')}

    # Create a new in-memory .mat file
    new_file = io.BytesIO()
    scipy.io.savemat(new_file, clean_data)

    # Reset pointer to start for reading
    new_file.seek(0)

    return new_file
