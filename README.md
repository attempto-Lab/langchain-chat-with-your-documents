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

You can either start a `pipenv shell` and use `python docchat_streamlit`
or run the following command:
```
export OPENAI_API_KEY=<YOUR_API_KEY>
pipenv run streamlit run docchat_streamlit_ui.py
```
