# Chat with your documents using LangChain

This repo contains the code for the blog articles "LangChain: Chat with your Documents!"
and "Interactive Web-App with Streamlit: Web-UI for Chat-with-your-Documents".

# Install Dependencies

## Install pipenv for user

```
python3 -m pip install --user pipenv
```

## Install Dependencies and create venv

```shell
pipenv install
```

### Run

You can launch a subshell in the virtual environment with `pipenv shell`:
```
pipenv shell

export OPENAI_API_KEY=<YOUR_API_KEY>
streamlit run docchat_streamlit_ui.py`
```

Alternatively you can use `pipenv run` to execute streamlit in the virtual environment:
```
export OPENAI_API_KEY=<YOUR_API_KEY>
pipenv run streamlit run docchat_streamlit_ui.py
```
