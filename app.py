# app.py

from flask import Flask, request, render_template, redirect, url_for, send_file, jsonify
from fetch_papers import fetch_papers
from summarize import summarize_text, extract_statistical_data
import plotly.express as px
import pandas as pd
import io
import logging
from geotext import GeoText
import networkx as nx
import plotly.graph_objects as go
from gensim import corpora  # type: ignore
from gensim.models import LdaModel  # type: ignore
from bertopic import BERTopic  # type: ignore
import nltk

# Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Import torch for device configuration
import torch

# Device configuration
if torch.cuda.is_available():
    device = torch.device('cuda')
    device_index = torch.cuda.current_device()
    print("Using CUDA device")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    device_index = 0  # MPS uses index 0
    print("Using MPS device")
else:
    device = torch.device('cpu')
    device_index = -1
    print("Using CPU")

# Load QA model with device configuration
from transformers import pipeline
qa_model = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad', device=device_index)

# Flask app initialization
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Global variables for storing data
app.config['PAPERS'] = []
app.config['SUMMARIES'] = []
app.config['YEARS'] = []
app.config['LOCATIONS'] = []
app.config['STAT_DATA'] = []
app.config['COAUTHOR_DATA'] = []
app.config['VENUES'] = []
app.config['CITATION_COUNTS'] = {}
app.config['LDA_TOPICS'] = []
app.config['BERT_TOPICS'] = []
app.config['FILTERS'] = {}
app.config['ABSTRACTS'] = []  # Store abstracts for QA

stop_words = set(stopwords.words('english'))

def preprocess_text(texts):
    processed_texts = []
    for text in texts:
        if text is None:
            processed_texts.append("")
            continue
        tokens = [word.lower() for word in text.split() if word.isalpha() and word.lower() not in stop_words]
        processed_texts.append(tokens)
    return processed_texts

def lda_topic_modeling(texts, num_topics=5):
    processed_texts = preprocess_text(texts)
    processed_texts = [text for text in processed_texts if text]

    if not processed_texts:
        return ["No valid abstracts for LDA topic modeling."]
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    topics = lda_model.show_topics(num_topics=num_topics, num_words=5, formatted=False)
    topic_results = []
    for idx, topic in topics:
        words = ", ".join([word for word, _ in topic])
        topic_results.append(words)
    return topic_results

def bert_topic_modeling(texts, min_docs=10):
    if len(texts) < min_docs:
        return ["Not enough documents for BERTopic modeling."]
    model = BERTopic()
    try:
        topics, _ = model.fit_transform(texts)
        topic_info = model.get_topic_info()[['Topic', 'Count', 'Name']].to_dict('records')
        return topic_info
    except ValueError as e:
        logging.error(f"BERTopic modeling failed: {e}")
        return ["BERTopic modeling encountered an error."]

@app.route('/')
def index():
    current_year = pd.Timestamp.now().year
    return render_template('index.html', current_year=current_year)

@app.route('/search', methods=['POST'])
def search():
    keyword = request.form['keyword']

    # Fetch papers
    papers = fetch_papers(keyword)

    # Limit the number of papers to 10
    max_papers = 10
    papers = papers[:max_papers]

    if not papers:
        return render_template('error.html', message="Failed to fetch papers. Please try again later.")

    summaries = []
    abstracts = []
    years = []
    locations = []
    stat_data = []
    coauthor_data = []
    venues = []
    citation_counts = {}

    for paper in papers:
        abstract = paper.get('abstract', None)
        title = paper.get('title', 'No Title')
        year = paper.get('year', 'Unknown')
        authors = paper.get('authors', [])
        url = paper.get('url', '#')
        venue = paper.get('venue', 'Unknown')
        citation_count = paper.get('citationCount', 0)
        publication_types = paper.get('publicationTypes', [])

        citation_counts[title] = citation_count

        if venue != 'Unknown':
            venues.append(venue)
        
        summary = summarize_text(abstract) if abstract else "No abstract available"

        processed_authors = []
        for author in authors:
            author_name = author.get('name', '')
            author_id = author.get('authorId', None)
            affiliations_list = []
            affiliations = author.get('affiliations', [])
            if affiliations:
                for affiliation in affiliations:
                    if isinstance(affiliation, str):
                        affiliations_list.append(affiliation)
                    elif isinstance(affiliation, dict):
                        affiliation_name = affiliation.get('name', '')
                        if affiliation_name:
                            affiliations_list.append(affiliation_name)
            processed_authors.append({
                'name': author_name,
                'authorId': author_id,
                'affiliations': affiliations_list
            })

        summaries.append({
            'title': title,
            'abstract': abstract,
            'summary': summary,
            'year': year,
            'url': url,
            'location': 'Unknown',
            'citation_count': citation_count,
            'publication_types': publication_types,
            'venue': venue,
            'authors': processed_authors
        })

        if abstract:
            abstracts.append(abstract)

        if year != 'Unknown':
            years.append(year)

        # Extract location
        for author in authors:
            affiliations = author.get('affiliations', [])
            if affiliations:
                for affiliation in affiliations:
                    affiliation_name = affiliation if isinstance(affiliation, str) else affiliation.get('name', '')
                    if affiliation_name:
                        places = GeoText(affiliation_name)
                        countries = places.country_mentions
                        if countries:
                            locations.extend(countries.keys())

        # Co-author data
        author_names = [author['name'] for author in authors if 'name' in author]
        if len(author_names) > 1:
            coauthor_data.append(author_names)

        # Statistical data
        stat = extract_statistical_data(abstract)
        if stat:
            logging.info(f"Extracted statistical data from paper '{title}': {stat}")
            stat_data.append({
                'title': title,
                'effect_size': stat['effect_size'],
                'ci_lower': stat['ci_lower'],
                'ci_upper': stat['ci_upper']
            })
        else:
            logging.info(f"No statistical data found in paper '{title}'.")

    # Topic modeling
    lda_topics = lda_topic_modeling(abstracts)
    bert_topics = bert_topic_modeling(abstracts)

    # Store data
    app.config['SUMMARIES'] = summaries
    app.config['YEARS'] = years
    app.config['LOCATIONS'] = locations
    app.config['STAT_DATA'] = stat_data
    app.config['COAUTHOR_DATA'] = coauthor_data
    app.config['VENUES'] = venues
    app.config['CITATION_COUNTS'] = citation_counts
    app.config['LDA_TOPICS'] = lda_topics
    app.config['BERT_TOPICS'] = bert_topics
    app.config['ABSTRACTS'] = abstracts
    app.config['FILTERS'] = {}

    # Generate all plots and data here and pass them to results.html
    locations = app.config['LOCATIONS']
    years = app.config['YEARS']
    stat_data = app.config['STAT_DATA']
    coauthor_data = app.config['COAUTHOR_DATA']
    citation_counts = app.config['CITATION_COUNTS']
    venues = app.config['VENUES']
    lda_topics = app.config['LDA_TOPICS']
    bert_topics = app.config['BERT_TOPICS']

    heatmap_json = None
    trend_json = None
    forest_plot_json = None
    coauthor_network_json = None
    venue_distribution_json = None
    citation_analysis_json = None

    if locations:
        df = pd.DataFrame(locations, columns=['country'])
        country_counts = df['country'].value_counts().reset_index()
        country_counts.columns = ['country', 'count']
        fig_heatmap = px.choropleth(country_counts, locations="country", locationmode='country names', color="count",
                                    title="Geographic Distribution of Research")
        heatmap_json = fig_heatmap.to_json()

    if years:
        df_years = pd.DataFrame(years, columns=['year'])
        df_years = df_years[df_years['year'] != 'Unknown']
        df_years['year'] = df_years['year'].astype(int)
        year_counts = df_years['year'].value_counts().reset_index()
        year_counts.columns = ['year', 'count']
        year_counts = year_counts.sort_values('year')
        fig_trend = px.bar(year_counts, x='year', y='count', title='Publication Trend Over Years')
        trend_json = fig_trend.to_json()

    if stat_data:
        df_stat = pd.DataFrame(stat_data)
        df_stat = df_stat.sort_values(by='effect_size')
        fig_forest = px.scatter(
            df_stat,
            x='effect_size',
            y='title',
            error_x=df_stat['ci_upper'] - df_stat['effect_size'],
            error_x_minus=df_stat['effect_size'] - df_stat['ci_lower'],
            labels={'effect_size': 'Effect Size', 'title': 'Study'},
            title='Forest Plot'
        )
        fig_forest.update_layout(yaxis={'categoryorder':'total ascending'})
        forest_plot_json = fig_forest.to_json()

    if coauthor_data:
        G = nx.Graph()
        for coauthors in coauthor_data:
            for i in range(len(coauthors)):
                for j in range(i+1, len(coauthors)):
                    G.add_edge(coauthors[i], coauthors[j])
        pos = nx.spring_layout(G, k=0.5)
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                showscale=False,
                color='blue',
                size=10,
                line_width=2))
        fig_coauthor = go.Figure(data=[edge_trace, node_trace],
                                 layout=go.Layout(
                                     title='Co-authorship Network',
                                     showlegend=False,
                                     hovermode='closest',
                                     margin=dict(b=20,l=5,r=5,t=40)
                                 ))
        coauthor_network_json = fig_coauthor.to_json()

    if venues:
        df_venues = pd.DataFrame(venues, columns=['venue'])
        venue_counts = df_venues['venue'].value_counts().reset_index()
        venue_counts.columns = ['venue', 'count']
        fig_venue = px.pie(venue_counts, names='venue', values='count', title='Publication Venue Distribution')
        venue_distribution_json = fig_venue.to_json()

    if citation_counts:
        df_citation = pd.DataFrame(list(citation_counts.items()), columns=['title', 'citation_count'])
        df_citation = df_citation[df_citation['citation_count'] > 0].sort_values(by='citation_count', ascending=False)
        df_citation['title'] = df_citation['title'].apply(lambda x: x[:50] + "..." if len(x) > 50 else x)
        fig_citation = px.bar(df_citation, y='title', x='citation_count', orientation='h', title='Citation Counts for Papers')
        citation_analysis_json = fig_citation.to_json()

    filters = app.config.get('FILTERS', {})
    current_year = pd.Timestamp.now().year
    return render_template('results.html', summaries=summaries, filters=filters, current_year=current_year,
                           heatmap_json=heatmap_json,
                           trend_json=trend_json,
                           forest_plot_json=forest_plot_json,
                           coauthor_network_json=coauthor_network_json,
                           venue_distribution_json=venue_distribution_json,
                           citation_analysis_json=citation_analysis_json,
                           lda_topics=lda_topics,
                           bert_topics=bert_topics)

@app.route('/download_citation_data')
def download_citation_data():
    citation_counts = app.config.get('CITATION_COUNTS', {})
    if not citation_counts:
        return render_template('error.html', message="No citation data available for download.")

    df = pd.DataFrame(list(citation_counts.items()), columns=['title', 'citation_count'])
    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    return send_file(output, mimetype='text/csv', as_attachment=True, download_name='citation_data.csv')

@app.route('/chat', methods=['POST'])
def chat():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    data = request.get_json()
    question = data.get('message', '')

    if not question:
        return jsonify({'answer': 'Please enter a question.'})

    abstracts = app.config.get('ABSTRACTS', [])

    if not abstracts:
        return jsonify({'answer': 'No abstracts available to answer your question.'})

    try:
        vectorizer = TfidfVectorizer().fit_transform([question] + abstracts)
        vectors = vectorizer.toarray()
        cosine_similarities = cosine_similarity([vectors[0]], vectors[1:]).flatten()
        most_relevant_index = cosine_similarities.argmax()
        context = abstracts[most_relevant_index]

        # Limit context length if necessary
        context = context[:2000]

        answer = qa_model({'question': question, 'context': context})
        return jsonify({'answer': answer['answer']})
    except Exception as e:
        logging.error(f"Error during QA: {e}")
        return jsonify({'answer': 'Sorry, an error occurred while processing your question.'})

if __name__ == '__main__':
    app.run(debug=True, port=5003)
