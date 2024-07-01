import pandas as pd
from transformers import BertJapaneseTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import numpy as np
import streamlit as st
import subprocess

# BERT日本語事前学習モデルの指定
model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'

# Mecab辞書の設定
subprocess.run(['python3', '-m', 'unidic.download', '--unidic-lite'])
subprocess.run(['pip', 'install', 'mecab-ipadic-nekodic'])

# トークナイザとモデルの読み込み
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# CSVデータの読み込み
df = pd.read_csv("train_data.csv", encoding='shift_jis')

# タイトルのエンベディング生成
def get_embeddings(titles):
    inputs = tokenizer(titles, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings

# タイトルのリスト
titles = df['タイトル'].tolist()
embeddings = get_embeddings(titles)

# t-SNEで2次元に次元削減
tsne_model = TSNE(perplexity=2, n_components=2, init='pca', random_state=23)
embeddings_tsne = tsne_model.fit_transform(embeddings)
df_embeddings = pd.DataFrame(embeddings_tsne, columns=['x', 'y'])

# Streamlitアプリケーションの構築
st.title('類似ニュース記事検索アプリ')
st.write('ニュース記事のタイトルを入力すると、類似記事を3つ表示します。')

# タイトルの入力
input_title = st.text_input('タイトルを入力してください')

if input_title:
    # 入力されたタイトルのエンベディングを生成
    input_embedding = get_embeddings([input_title])
    
    # 類似度計算
    similarities = cosine_similarity(input_embedding, embeddings)[0]
    
    # 類似度が高い上位3つのインデックスを取得
    top_indices = np.argsort(similarities)[-3:][::-1]
    
    # 結果表示
    for idx in top_indices:
        similarity_percent = similarities[idx] * 100
        st.write(f'**タイトル**: {df.loc[idx, "タイトル"]}')
        st.write(f'**課題**: {df.loc[idx, "課題"]}')
        st.write(f'**サービス内容**: {df.loc[idx, "サービス内容"]}')
        st.write(f'**URL**: [記事リンク]({df.loc[idx, "URL"]})')
        st.write(f'**類似度**: {similarity_percent:.2f}%')
        st.write('---')