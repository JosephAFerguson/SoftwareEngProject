The Python ML model. Currently, we are using Anaconda as the virtual environment. After installing Anaconda, go
to the *_py/_* directory and run the following command:

`conda create -p ./.conda python=3.13 &&
conda activate ./.conda &&
conda install -c conda-forge scikit-learn &&
pip install -r requirements.txt`

To test that this all successful, you can run this command:

`python -c "import flask, torch, sklearn; print(f'Flask: {flask.__version__}\nPyTorch: {torch.__version__}\nSklearn: {sklearn.__version__}')"`

