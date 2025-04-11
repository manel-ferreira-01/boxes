import scipy.io
import io

def codigo(file_pointer):
    """
    Reads all variables from a MATLAB .mat file given a file pointer,
    writes them to a new in-memory .mat file, and returns the new file pointer.
    
    Parameters:
    file_pointer (file-like object): Opened .mat file in binary read mode.

    Returns:
    new_file_pointer (io.BytesIO): In-memory file-like object containing the cloned .mat file.
    """
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
