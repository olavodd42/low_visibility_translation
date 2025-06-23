FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

# Install JupyterLab and ML libraries
RUN pip install --no-cache-dir \
      jupyterlab \
      transformers datasets \
      accelerate \
      torchaudio \
      matplotlib seaborn \
      pandas numpy

WORKDIR /workspace
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]