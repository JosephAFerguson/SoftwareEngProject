Installations needed: Node.js

Modules needed - Frontend: react-router-dom
`npm install react-router-dom`

Modules needed - Backend: flask, torch, skikit-learn
  need to create a virtual env for this most likely
  Here it is with anaconda already installed
    `conda create -n sfteng python=3.13`
    `conda activate sfteng`
    with the venv activated, the following:
      `conda install -c conda-forge scikit-learn`
      `pip install Flask`
      `pip install flask flask-cors torch`
      `pip install torch torchvision torchaudio`
      to test in the terminal run:
      `python -c "import flask, torch, sklearn; print(f'Flask: {flask.__version__}\nPyTorch: {torch.__version__}\nSklearn: {sklearn.__version__}')"`

  You will need to select this venv in the project in VsCode
  
  Database to be determined

