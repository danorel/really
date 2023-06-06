# Foobar 

### Prerequisite 

- Python: 3.10
- Jupyter Notebook: 6.4.12

### Environment

1. Install python virtual environment:

```shell
python3 -m venv env 
```

2. Activate python virtual env

```shell
source env/bin/activate
```

3. Install python dependencies from requirements.txt:

```shell
pip3 install -r requirements.txt
```

4. Setup `pre-commit` to make hook with `.git`:

```shell
pre-commit install
pre-commit run --all-files
```

5. Setup Jupyter Notebook virtual env

```shell
python3 -m ipykernel install env
```

### Vim/NeoVim users:

6. Install pynvim to enable coc:
```shell
python3 -m pip install --upgrade pynvim
```
