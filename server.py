from flask import Flask
from flask import request
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from flask_cors import CORS

from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model='gpt-4-1106-preview')
vectorstore = Chroma(persist_directory='./chroma_db', embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

template = """You are a helpful real estate agent. Use the following retrieved home listing infos to generate useful insight for home suggestions. 
If can't think of a good answer, just say that you don't know. Use ten sentences maximum to summarize the homes and keep the answer concise.  
Provide a link for each home in the retrieved context.  Start with something like "Here are some <number of bedrooms> homes in <location> that might interest you"

Description: {} 

Listing Infos: {} 

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

def chain(question):
	context = retriever.invoke(question)
	final_prompt = template.format(question, context)
	msg = model.invoke(final_prompt)
	return msg

app = Flask(__name__)
CORS(app)

@app.post('/flagler')
def flagler():
    body = request.get_json()
    if (body['query'] == None):
      return '''<p>Could not find query in your request</p>'''
    query = body['query']
    response = chain(query).content
    print(response)
    return { 'suggestions': response }
    
app.run(host="0.0.0.0")