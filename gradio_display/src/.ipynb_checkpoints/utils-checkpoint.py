import tempfile
import csv
from collections.abc import Iterable
from collections import OrderedDict
import os
import cv2
import json

def flatten(item):
    """Recursively flattens nested lists or tuples, treats strings and dicts as atomic."""
    if isinstance(item, dict):
        for value in item.values():
            yield from flatten(value)
    elif isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
        for sub in item:
            yield from flatten(sub)
    else:
        yield item

def all_dicts_with_same_keys(data_list):
    """Check if all items are dicts and have the same keys."""
    if not all(isinstance(item, dict) for item in data_list):
        return False
    first_keys = list(data_list[0].keys())
    return all(list(d.keys()) == first_keys for d in data_list)

def write_list_to_temp(data_list,prefix="track_",suffix=".csv"):

    if suffix==".json":
        with tempfile.NamedTemporaryFile(mode='wt', delete=False, prefix=prefix, suffix=suffix) as temp_file:
            temp_file.write(data_list)
            return os.path.basename(temp_file.name),temp_file.name 


    """Writes a list of data to a temporary CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', newline='', delete=False, prefix=prefix, suffix=suffix) as temp_file:
        writer = csv.writer(temp_file)

        # If all elements are dicts with same keys, write header
        if all_dicts_with_same_keys(data_list):
            header = list(data_list[0].keys())
            writer.writerow(header)
            for d in data_list:
                flat_row = list(flatten(d))
                writer.writerow(flat_row)
        else:
            # For non-uniform data
            for element in data_list:
                if isinstance(element, dict):
                    flat_row = list(flatten(element))
                elif isinstance(element, Iterable) and not isinstance(element, (str, bytes)):
                    flat_row = list(flatten(element))
                else:
                    flat_row = [element]
                writer.writerow(flat_row)

        return os.path.basename(temp_file.name),temp_file.name 

#------------------- UTILS -------------------    

def flatten2(item):
    """Recursively flattens nested iterables except strings and bytes."""
    if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
        for sub in item:
            yield from flatten(sub)
    else:
        yield item

def write_list_to_temp_csv2(data_list,prefix="track_"):
    """Writes a list to a temp CSV file, flattening each item if needed."""
    with tempfile.NamedTemporaryFile(mode='w', newline='', delete=False, prefix=prefix,suffix='.csv') as temp_file:
        writer = csv.writer(temp_file)
        for element in data_list:
            # Wrap non-iterable elements in a list so they're still written as a row
            if isinstance(element, Iterable) and not isinstance(element, (str, bytes)):
                flat_row = list(flatten(element))
            else:
                flat_row = [element]
            writer.writerow(flat_row)
        return temp_file.name

def getfromkey(lista,key):
    for l in lista:
        if key in l:
            return l[key]
    return None

def getrowsfromjson(seqdetections):
    rows=[]
    for i,img in enumerate(seqdetections) :
        for obj in img:
            if obj:
                tmp=list(obj.values())
                tmp.insert(0,i+1)
                rows.append(flatten(tmp))
    return rows

def getdictrowsfromjson(seqdetections):
    rows=[]
    for i,img in enumerate(seqdetections) :
        for obj in img:
            if obj:
                obj.update({"frame":i+1})
                rows.append(obj)
    return rows
    
# Define a function to resize an image
def force_resize_image(img,max_size):
    # Resize the image so that it has a maximum dimension of 640 pixels
    if img.shape[0]>img.shape[1]:
        img = cv2.resize(img, (int(max_size *img.shape[1]/img.shape[0]),max_size), interpolation=cv2.INTER_LINEAR)
    else:
        img = cv2.resize(img, (max_size,int(max_size *img.shape[0]/img.shape[1])), interpolation=cv2.INTER_LINEAR)
    
    # Return the resized image as a PIL Image object
    return img
 