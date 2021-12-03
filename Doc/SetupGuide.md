# Python Project Setup

***Author: Dingkun***

***Visit [My GitHub](https://github.com/Oct19) for more info***

---

## Git initialize

- Clone repository from Link

If .gitignore is not working:

    git rm -rf --cached .
    git add .

## Install specific Python version

1. Download Python 3.8.0, 64-bit
2. Install the Python executable. I recommend a custom installation. There’s no need to add it to PATH.

## Virtual environment

### Create VEnv (Using virtualenv)

In work folder, open terminal:

    py -m virtualenv -p="C:\Program Files\Python38\python.Exe" .virtenv
    .virtenv/Scripts/activate.bat

Python: Select interpreter from `.virtenv/Scripts/python.exe`

---

### Install required packages

Option 1: Install from requirement file:

    pip install -r requirements.txt

Option 2: Manually install packages with pip

    python -m ensurepip
    pip install numpy

To Install packages with user rights

    pip install numpy --user

To Update packages

    pip install -U numpy

## Jupyter (Optional)

Install Jupyter from VS Code Extensions.

If Jupyter cell fail to run, try:

    python -m pip install 'traitlets==4.3.3' --force-reinstall
