# PyTorch Flask Sentiment Analysis API

<br>

![Screenshot-20210106221757-384x410](https://user-images.githubusercontent.com/40786348/103772526-0924dc00-506d-11eb-9bf8-7c6f2d81f635.png)

#### Check the demo [here](http://43a0b71b5aa4.ngrok.io)

<br>

---

<br>


## Details

- Sentiment Analysis Using AI ([LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) modeling based on [Fasttext](https://fasttext.cc/) Embedding)
- Model was trained using [Naver Movie Review Data](https://github.com/e9t/nsmc)


<br>

## Requirements

Create new environment using following command:

    conda create -n nlp python=3.8 
    conda activate nlp

Install them from `requirements.txt`:

    pip install -r requirements.txt

<br>

## Local Deployment

Run the server:

    python app.py

<br>

## Next Step

- The endpoint **/predict** assumes input text to be always Korean. This may not hold true for all requests. English input text is not allowed at the moment.
- We run the Flask server in the development mode, which is not suitable for deploying in production. You can check out this [tutorial](https://flask.palletsprojects.com/en/1.1.x/tutorial/deploy/) for deploying a Flask server in production.
- Could modify service to be able to return predictions for multiple sentences at once.
- Lightweight model for more reduced inference time.
- Flask async task handling using [Celery](https://docs.celeryproject.org/en/stable/)

<br>

## License

The mighty MIT license. Please check `LICENSE` for more details.