"""
Módulo de procesamiento de imágenes.
@author Carlos Soto Pérez <carlos348@outlook.com>
"""
import boto3
import json
import os
import time
import cv2
from flask import Flask, render_template, g, make_response, redirect, request

BUCKET_NAME = 'stereo-cloud-o2020'
app = Flask(__name__)

def get_s3_client():
    if 's3_client' not in g:
        g.s3_client = boto3.client('s3')
    return g.s3_client

def get_disparity(left, right, max_disp=-1, wsize=-1):
    if max_disp <= 0 or max_disp%16 != 0:
        raise Exception('Incorrect max_disparity value: it should be positive and divisible by 16')
    if wsize <= 0 or wsize%2 != 1:
        raise Exception('Incorrect wsize value: it should be positive and odd')

    max_disp/=2
    max_disp = int(max_disp)
    if max_disp%16 != 0:
        max_disp += 16-(max_disp%16)

    left_for_matcher = None
    right_for_matcher = None
    width = int(left.shape[1] * 0.5)
    height = int(left.shape[0] * 0.5)
    dim = (width, height)
    left_for_matcher = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_for_matcher = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoBM.create(numDisparities=max_disp, blockSize=wsize)

    disparity = stereo.compute(left_for_matcher, right_for_matcher)
    return disparity

@app.route('/process_img', methods=['POST'])
def process_img():
    body = request.get_json()
    left = body['left']
    right = body['right']
    path_left = '/app/' + left
    path_right = '/app/' + right
    s3 = get_s3_client()
    # Crear imagen left
    open(path_left, 'a').close()
    with open(path_left, 'r+b') as f:
        s3.download_fileobj(BUCKET_NAME, left, f)

    # Crear imagen right
    open(path_right, 'a').close()
    with open(path_right, 'r+b') as f:
        s3.download_fileobj('stereo-cloud-o2020', right, f)
    
    l = cv2.imread(path_left)
    r = cv2.imread(path_right)
    disparity = get_disparity(l, r, max_disp=16, wsize=15)
    os.remove(path_left)
    os.remove(path_right)

    out_name = time.time_ns() + '.png'
    out_path = '/app/' + out_name
    cv2.imwrite(out_path, disparity)
    data = open(out_path, 'r+b')
    s3.put_object(Bucket=BUCKET_NAME, Body=data, Key=out_name)
    os.remove(out_path)
    response = {
        'key': out_name
    }

    return response