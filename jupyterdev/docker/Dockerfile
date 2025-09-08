FROM quay.io/jupyter/scipy-notebook
USER root
RUN apt -y update 
RUN apt -y install \
      less \
      gnuplot \
      libnetcdf-dev \
      octave \
      git \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

USER jovyan

# Install requirements
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose port 8061 according to ai4eu specifications
EXPOSE 8061 8888 7860

#WORKDIR ${WORKSPACE}

#CMD ["jupyter","lab","--allow-root"]
#CMD ["start-notebook.py", "--NotebookApp.token='jupyter']