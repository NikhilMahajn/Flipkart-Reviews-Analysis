from bs4 import BeautifulSoup
import requests
import re
import string
import json
import pandas as pd
from flask import Flask,render_template,request,jsonify
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
from scipy.sparse import hstack
nltk.download('stopwords')
stpwords = set(stopwords.words('english'))  # Convert stopwords list to a set for fast lookup
stem = PorterStemmer()

with open('models/sentiment_model2.pkl','rb') as f:
	model, vectorizer_review, vectorizer_summary = pickle.load(f)

app = Flask(__name__)

@app.route('/',methods = ['GET'])
def index():
	return render_template('index.html')

@app.route('/generate_reviews',methods=['POST'])
def generate_reviews():
	try:
		print(request.content_type)
		if request.content_type != 'application/json':
			return jsonify({"error": "Unsupported Media Type, expecting JSON"}), 415
		data = request.get_json()
		if not data or 'productLink' not in data:
			return jsonify({"error": "Invalid JSON data"}), 400
		link = data['productLink']
		dataset = fetch_reviews(link)
		data = dataset.copy()
		data['Review'] = data['Review'].apply(lambda X:re.sub(r'READ MORE','',X))
		data['Summary'] = data['Summary'].apply(lambda X:re.sub(r'READ MORE','',X))
		# dataset.to_csv('data//reviews_flipkart.csv',index=False)

		dataset['Review'] = dataset['Review'].apply(clean)
		dataset['Summary'] = dataset['Summary'].apply(clean)

		review_vector = vectorizer_review.transform(dataset['Review'])
		summary_vector = vectorizer_summary.transform(dataset['Summary'])

		reviews_filtered = hstack([review_vector, summary_vector])

		predictions_review = model.predict(reviews_filtered)
		count = predictions_review.shape[0]
		positive_count = sum(predictions_review)
		negative_count = count - positive_count
		print(positive_count,negative_count)

		postive_percentage = round((positive_count/count)*100)
		negative_percentage = round((negative_count/count)*100)
		print(postive_percentage,negative_percentage)

		return jsonify({'reviews': data['Review'].tolist(),
				   		'summary': data['Summary'].tolist(),
						'predictions': predictions_review.tolist(),
						'positive_percentage': postive_percentage,
						'negative_percentage': negative_percentage,
						})
	except Exception as e:
		return jsonify({"error": str(e)}), 500



def fetch_reviews(link):
	review_link = re.sub(r'/p/','/product-reviews/',link)
	review = []
	summary = []
	c = 0
	while c<10:
		soup = BeautifulSoup(requests.get(review_link).content, 'html.parser')
		print('page')
		review_block = soup.find_all('div',"col EPCmJX Ma1fCG")
		for i in review_block:
			rvsoup = BeautifulSoup(str(i), 'html.parser')
			review.append(rvsoup.find('p','z9E0IG').text)
			summary.append(rvsoup.find('div','ZmyHeo').text)
			next_link = soup.find_all('a','_9QVEpD')[-1].get('href')
		if not next_link:
			break
		else:
			review_link = 'https://www.flipkart.com'+next_link
		c +=1
	data = pd.DataFrame({'Review':review,'Summary':summary})
	return data

def clean(i):

    i = re.sub(r'READ MORE', '', i)
    i = i.lower()  # Convert to lowercase
    words = word_tokenize(i)  # Tokenize sentence
    words = [stem.stem(word) for word in words if word not in stpwords]
    words = [word for word in words if word not in string.punctuation]
    return ' '.join(words)  # Join words back into a sentence
    


if __name__ == '__main__':
	app.run(debug=False,host='0.0.0.0',port=5000)

