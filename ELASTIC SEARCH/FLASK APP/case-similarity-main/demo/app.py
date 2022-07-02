# Copyright 2015 IBM Corp. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from flask import Flask, render_template, request
import json
import requests
import random
import traceback
import collections
from typing import Dict, List
import elasticsearch
import pandas as pd
from elasticsearch import Elasticsearch
import re

app = Flask(__name__)


# initializing the instance of elasticsearch to localhost at port 9200(default)
es = Elasticsearch(HOST="http://localhost", PORT=9200)
# create the index

es.indices.delete(index='judgement_index', ignore=[400, 404])
es.indices.create(index="judgement_index")

# load the json file

file = open("data_for_index_p_combined.json", "r", encoding="utf-8")

data = json.load(file)
file.close()
# insert the data in the index

for i in data.keys():
  es.index(index="judgement_index", doc_type="cases", id=int(i), body=data[i])


def get_id(link):
  regex = re.compile(r'[0-9]+')
  result = regex.search(link)

  if (result):
    return str(result.group())
  else:
    return ""

@app.route('/')
def Welcome():
  return app.send_static_file('index.html')

def return_similar(docid):
  file = pd.read_csv("casemine.csv")

  titles = []

  for i in range(file.shape[0]):
    res = []
    if get_id(str(file.iloc[i, 1])) == str(docid):
      #append doc id of case in which keyword was found
      res.append(docid)
      #append its name
      res.append(file.iloc[i, 0])
      #append its link
      res.append(file.iloc[i, 1])
      #append its similar judgement
      res.append(file.iloc[i, 2])
      #append its similar judgement link
      res.append(file.iloc[i, 3])
    if get_id(str(file.iloc[i, 3])) == str(docid):
      #append doc id of case in which keyword was found
      res.append(docid)
      #append its name
      res.append(file.iloc[i, 2])
      #append its link
      res.append(file.iloc[i, 3])
      res.append(file.iloc[i, 0])
      res.append(file.iloc[i, 1])
      res.append(-100)
    if len(res)>0:
      titles.append(res)
  return titles


def index_func(text_to_search):
    """
    parameters:
    where_to_search: scope of search eg: title, author, etc.
    text_to_search: text that you are looking for in the document

    output:
    1. returns, if no output is found
    2. details of judgement, otherwise
    """

    # store the result of the query
    body = {
      "_source": [
        "title",
        "author"
      ],
      "size": 10000,
      "query": {
        "multi_match": {
          "query": text_to_search,
          "type": "phrase",
          "fields": ["author", "title", "content"]
        }
      }

    }

    result = es.search(index="judgement_index", doc_type="cases", body=body)

    # extract the number of results from the result object
    num_results = result['hits']['total']['value']



    results = []
    # return if nothing is found
    if num_results == 0:
      #print("No result found!")
      return
    else:
      #print the number of results found
      #print(str(num_results) + " result/s found!\n\n")

      # print the details of the judgement
      for i in range(num_results):
        res = result['hits']['hits'][i]['_source']
        score = result['hits']['hits'][i]['_score']
        docid = result['hits']['hits'][i]['_id']
        #print("RESULT " + str(i + 1))
        #print("---------")
        #print("Document title : " + str(res['title']))
        #print("Document author : " + str(res['author']))
        #print("Score : " + str(score))
        similar_judgements = return_similar(docid)
        #print("SIMILAR JUDGEMENTS")
        for arr in similar_judgements:
          results.append(arr)

      return results


@app.route('/similar_case', methods=['POST'])
def similar_case():
  input_text = request.form.get('input_text')


  try:
    output = "<b>Input Text</b><br/><br/>"
    output += "<p>" + input_text + "</p><br/>"
    results = None
    debug_message = None
    try:
      results = index_func(input_text)
    except Exception as e:
      output += "<br/>Error: " + str(e) + "<br/>"
      return output

    if results is None:
      output += "<p>Error. Try again</p>"
      output += "<p>" + debug_message + "</p>"
      return output

    if len(results) < 1:
      output += "<p>No results.</p>"
      output += "<p>" + debug_message + "</p>"
      return output

    pruned_results = []
    output += "<table border=2>"
    output += "<tr>"
    output += "<td width=\"10%\">" + "<b>Sno</b>" + "</td> "
    output += "<td width=\"10%\">" + "<b>Case ID</b>" + "</td> "
    output += "<td width=\"30%\">" + "<b>Keyword found in</b>" + "</td> "
    output += "<td width=\"30%\">" + "<b>Similar Judgement</b>" + "</td> "
    output += "</tr>"
    print(results)

    """
    for k,v in results.items():
      if 'debug' in k:
        debug_message = k['debug']
        continue
    """

    """
    #sorting the results
    for i in range(len(results)):
      for j in range(i):
        if results[i][-1] < results[j][-1]:
          results[i] , results[j] = results[j] , results[i]
    """


    counter = 0
    for res in results:
      counter = counter + 1
      output += "<tr>"
      output += "<td width=\"15%\">" + str(counter) + "</td> "
      output += "<td width=\"15%\">" + str(res[0]) + "</td> "
      if len(str(res[2])) > 3:
          output += "<td width=\"15%\"><a href=\"" + str(res[2]) + "\">" + str(res[1]) +"</a></td>"
      else:
          output += "<td width=\"15%\">"+str(res[1])+"</td>"
      if len(str(res[4])) > 3:
          output += "<td width=\"15%\"><a href=\"" + str(res[4]) + "\">" + str(res[3]) +"</a></td>"
      else:
          output += "<td width=\"15%\">"+str(res[3])+"</td>"
      output += "</tr>"
    output += "</table>"
    print(output)
  except Exception as e:
    traceback.print_exc()

  return output

port = os.getenv('PORT', '5000')

if __name__ == "__main__":
  app.run(host='0.0.0.0', port=int(port))
