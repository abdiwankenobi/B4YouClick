import argparse
import sys
import time
import io 
import base64
import timeit
from timeit import default_timer
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types
from google.cloud.gapic.videointelligence.v1beta1 import enums
from google.cloud.gapic.videointelligence.v1beta1 import (
    video_intelligence_service_client)
from google.cloud import storage

#imports for wordnet and iteratians
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from itertools import cycle
from itertools import product

import pafy 
from textblob import TextBlob
from fuzzywuzzy import fuzz

#function uploads videos to Google Cloud Storage from local
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))
    
#function initiates video processing using the Video Intelligence API by Google and collects data
def analyze_labels(path):
    """ Detects labels given a GCS path. """
    video_client = (video_intelligence_service_client.
                    VideoIntelligenceServiceClient())
    features = [enums.Feature.LABEL_DETECTION, enums.Feature.SHOT_CHANGE_DETECTION]
    operation = video_client.annotate_video(path, features)
    print('\nProcessing video for label annotations:')

    while not operation.done():
        sys.stdout.write('.')
        sys.stdout.flush()
        time.sleep(20)

    print('\nFinished processing.')
    

    results = operation.result().annotation_results[0]
    return results
	
#compare video tags with title
def compareTitle(tags, title):
    if(len(title)!= 0):
        text = " ".join(title)     #create text string
        title_set= [word for word in title_words if word not in stopwords.words('english')] #remove stopwords
        title_set= set(title_set) # remove duplicates
        #initialise variables
        tags_one = [] 
        tags_multi = []
        match = 0
        #split video tags one and multi
        for t in tags:
            if(len(t.split())==1):
                tags_one.append(t)
            else:
                tags_multi.append(t)
        #compare one word tags with title words
        for t in tags_one:
            for ti in title_set:
                check = False
                for s in wordnet.synsets(t):
                    lemmas = s.lemmas()
                    for l in lemmas:
                        if (fuzz.token_sort_ratio(l.name(), ti)>90):
                            #l.name() == ti:
                            check = True
                if check == True:
                    match+=1
        #create ngamrs and compare multi words
        for t in tags_multi:
            check = False
            blob = TextBlob(text).ngrams(n=len(t.split()))
            for b in blob:
                if(t == " ".join(b)):
                    check=True

            if check == True:
                match+=1
        return match*100/len(title_set)
    else:
        return 0

#compare video tags with keywords
def compareKeyword(tags, keywords):
    if(len(keywords) != 0):
        match=0 # create variable
        for t in tags:
            for k in keywords:
                check = False
                if(len(t.split()) == 1): #if label contains 1 word
                    for s in wordnet.synsets(t): #get synstet 
                        lemmas = s.lemmas()   # get synonyms 
                        for l in lemmas:
                            if (fuzz.token_sort_ratio(l.name(), k)>90): #if word is similar to keyword
                                check = True

                else: # if word contain more than 1 word
                    if(fuzz.token_sort_ratio(t, k)>90): # if multiword is similar to keyword 
                        check=True

                if check == True:
                    match+=1
        return match*100/len(keywords) #get percentage
    else:
        return 0
def compareThumb(tags, thumb_labels):
    if(len(thumb_labels) != 0):
        match=0
        for t in tags:
            for th in thumb_labels:
                if(t == th):
                    match = match+1
        return match*100/len(thumb_labels)
    else:
        return 0
		
#Initiating process with sample Youtube Url collecting its data including the video itself
url="https://www.youtube.com/watch?v=-ugJZhL-cbc"

video = pafy.new(url)
best = video.getbest(preftype="mp4")
title =video.title
description = video.description
keywords = video.keywords

thumbnail = "http://img.youtube.com/vi/"+video.videoid+"/maxresdefault.jpg"

#if file exists already delete and download the video
try:
    os.remove("C:/College/MSC/Project/Analysis/downloads/file3.mp4")
except OSError:
    pass
best.download(filepath="C:/College/MSC/Project/Analysis/downloads/file3."+best.extension)

#upload downloaded video to Google Cloud Storage
upload_blob('isentropic-rush-4581', 'C:/College/MSC/Project/Analysis/downloads/file3.mp4', 'file3')

#Start processing uploaded video
results = analyze_labels('gs://isentropic-rush-4581/file3')
  
#obtain labels/tags from results
tags = []

for i in results.label_annotations:
    tags.append(i.description)
    
tags = [item.lower() for item in tags]


#compare Keywords
keywords = [item.lower() for item in keywords]
keywords_compared = compareKeyword(tags, keywords)


#compare thumbnail
client = vision.ImageAnnotatorClient()
image = types.Image()
image.source.image_uri = thumbnail
response = client.label_detection(image=image)
labels = response.label_annotations

#creating new list holding the labels only
thumb_labels=[]
for label in labels:
    thumb_labels.append(label.description)

thumb_labels = [item.lower() for item in thumb_labels]

#if thumbnail is not valid, video has no Thumbnail image, return 0
if not thumb_labels:
    thumb_compared=0
else:
    thumb_compared=compareThumb(tags, thumb_labels)
print(thumb_compared)

title_words= TextBlob(video.title+" "+ video.description).correct().lower().split()
title_words = [word for word in title_words if word.isalpha()]
title_words = [word for word in title_words if word not in stopwords.words('english')]
title_compared = compareTitle(tags, title_words)


total = len(keywords)+len(set(title_words))+len(thumb_labels)
finalresult= len(keywords)/total*keywords_compared+len(set(title_words))/total*title_compared+len(thumb_labels)/total*thumb_compared

value_list = [keywords_compared, thumb_compared, title_compared, len(keywords), len(thumb_labels), len(title_words), total, len(tags), finalresult]
print(value_list)