from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import shutil
import yaml

from utils import draw_bbox, analysis_bbox, StatAnalysis

app = Flask(__name__)

# 사진이 저장된 폴더들
FOLDERS = {
    'house': 'static/house',
    'tree': 'static/tree',
    'person': 'static/person'
}

# 통계자료 데이터
STAT_DATA = {
    'house': './stat_house.json',
    'tree': './stat_tree.json',
    'person': './stat_person.json',
}

stat = StatAnalysis(STAT_DATA)

# 처리된 이미지를 저장할 폴더
PROCESSED_FOLDER = 'static/processed'

@app.route('/')
def gallery():
    # 각 폴더에서 이미지 목록 가져오기
    images = {}
    for folder_name, folder_path in FOLDERS.items():
        if os.path.exists(folder_path):
            images[folder_name] = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
        else:
            images[folder_name] = []
    return render_template('index.html', images=images)

@app.route('/process', methods=['POST'])
def process_images():
    # 처리된 이미지 폴더 초기화
    if os.path.exists(PROCESSED_FOLDER):
        shutil.rmtree(PROCESSED_FOLDER)
    os.makedirs(PROCESSED_FOLDER)

    # 폼 데이터 가져오기
    house_img = request.form.get('house_img')
    tree_img = request.form.get('tree_img')
    person_img = request.form.get('person_img')
    age = request.form.get('age')
    gender = request.form.get('gender')
    
    # 선택된 이미지 경로 
    selected_images = { 
        'house': os.path.join(FOLDERS['house'], house_img),
        'tree': os.path.join(FOLDERS['tree'], tree_img),
        'person': os.path.join(FOLDERS['person'], person_img)
    }

    # 처리된 이미지 파일명
    processed_images = []
    inference_result = {} 
    # 각 이미지를 OpenCV로 처리
    for folder, img_path in selected_images.items():
        if not os.path.exists(img_path):
            continue

        # 이미지 읽기
        img = draw_bbox(img_path)        
        if img is None:
            continue

        inference_result.update(analysis_bbox(img_path)) 

        # 처리된 이미지 저장
        output_filename = f"{folder}_processed.jpg"
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)
        cv2.imwrite(output_path, img)
        processed_images.append(output_filename)

    # 이미지 목록 다시 로드
    images = {}
    for folder_name, folder_path in FOLDERS.items():
        if os.path.exists(folder_path):
            images[folder_name] = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
        else:
            images[folder_name] = []

    # 통계 분석
    house_graphs = stat.drawGraphs(age, gender, inference_result, processed_images)
    # 결과지 텍스트
    output_message = stat.get_text_result(age, gender, inference_result)
    
    output_message.append('')

    return render_template('index.html', images=images, processed_images=processed_images, message=output_message)

if __name__ == '__main__':
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port)