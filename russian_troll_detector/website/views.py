# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import json
from django.shortcuts import render
from annoying.decorators import render_to
# Create your views here.
from django.http import HttpResponse
from django.template import loader
from .tasks import get_tweets, calculate_features

from sklearn.externals import joblib
from .extractors import ColumnExtractor, TextExtractor

""" CODE FOR LOADING MODEL """
import sys
sys.path.append('website')
model_name = 'FAKE_MODEL.pkl'
clf = joblib.load('website/' + model_name)

def index(request):
    template = loader.get_template('website/index.html')
    context = {}
    return HttpResponse(template.render(context, request))


def predict(request):
    data = ''
    if request.is_ajax():
        print('is ajax')
        username = request.GET.get('username', 'None')
        print('username is ' + username)

        # 1.get tweets
        try:
            user_id, author_tweets, follower, following = get_tweets(username)
        except TypeError:
            data = {'is_rt': '0', 'prob': ' user is verified'}
            json_data = json.dumps(data)
            return HttpResponse(json_data, content_type='application/json')
        # 2.calculate features needed for prediction
        X_test = calculate_features(user_id, author_tweets, follower, following)
        # 3.get prediction and probability of being RT
        is_rt = float(clf.predict(X_test)[0])
        prob = float(clf.predict_proba(X_test).tolist()[0][1]*100)

        data = {'is_rt': is_rt, 'prob': prob}

        for label, content in X_test.iteritems():
            for index, value in content.iteritems():
                if(label == 'text'):
                    data[label] = value
                else:
                    data[label] = float(value)

    else:
        data = {'response': 'Not an ajax request'}

    # for status in data:
        # print(status)
        # data_dict["ID"] = status.id
        # data_dict["ScreenName"] = status.user.screen_name

    #print(data)
    json_data = json.dumps(data)
    return HttpResponse(json_data, content_type='application/json')


@render_to('website/details.html')
def visualize_account_details(request):
    template = loader.get_template('website/details.html')
    context = {}
    return HttpResponse(template.render(context, request))