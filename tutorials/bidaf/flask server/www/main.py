from flask import Flask, render_template, request
import numpy as np
import torch
from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
import requests
import urllib
from bs4 import BeautifulSoup
import pdb

app = Flask(__name__)
archive = load_archive("bidaf.tar.gz", cuda_device=0)
predictor = Predictor.from_archive(archive)


def google_search(text):
    "Takes raw text as input"
    text = urllib.parse.quote_plus(text)
    url = 'https://google.com/search?q=' + text
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "lxml")

    get_details = soup.find_all("div", attrs={"class": "g"})
    final_data = []
    for details in get_details:
        link = details.find_all("h3")
        # links = ""
        for mdetails in link:
            links = mdetails.find_all("a")
            lmk = ""
            for lnk in links:
                lmk = lnk.get("href")[7:].split("&")
                sublist = []
                sublist.append(lmk[0])
            final_data.append(sublist)

    # choose the first url
    url = final_data[0][0]
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # extract paragraphs
    query = ''
    for i, p in enumerate(soup.find_all("p")):
        if i > 10:
            break
        query += p.text

    return query, url


@app.route('/')
def student():
    return render_template('index.html', output="")


@app.route('/result', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':

        # Search google for results
        query = request.form['Input']
        passage, url = google_search(query)

        results = predictor.predict(
            passage=passage,
            question=query
        )

        output = results['best_span_str']
        return render_template("index.html", output=output)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
