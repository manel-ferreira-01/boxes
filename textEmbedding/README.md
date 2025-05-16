# Generate Text Embedding using sentence-transformers

Box that encodes text strings 
'''python

    # SPECIFIC CODE STARTS HERE

    text_list=mat_data["text"]
    list_embeddings = get_embeddings(text_list)

    # SPECIFIC CODE ENDS HERE

    f=io.BytesIO()
    # WRITE RETURNING DATA
    savemat(f,{"embeddings":list_embeddings})


Input .mat file must have one variable "text" and returns a dictionary with key "embeddings" containing the string embeddings
