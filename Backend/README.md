# Setup for Local Backend Environment

> As of now we are using Python for the entire backend. Ideally, I (Garrett) would like to use this
> project as an excuse to try new technologies and use something like Go for the backend. Regardless, 
> we will have have to use Python for all the ML models, so we still need this Python environment.

First, we will be using Anaconda as the virtual environment. After installing Anaconda, go to the backend
directory and run the following command:

`conda create -p ./.conda python=3.13 &&
conda activate ./.conda &&
conda install -c conda-forge scikit-learn &&
pip install -r requirements.txt`

To test that this all successful, you can run this command:

`python -c "import flask, torch, sklearn; print(f'Flask: {flask.__version__}\nPyTorch: {torch.__version__}\nSklearn: {sklearn.__version__}')"`

Once we decide on a database and if we decide to use a different language for the backend middleware layer, those instructions
will be added here.
