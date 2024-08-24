from django.shortcuts import render
from youtube_transcript_api import YouTubeTranscriptApi # type: ignore
import nltk
from nltk.corpus import stopwords
from youtube_transcript_api.formatters import TextFormatter
from nltk.tokenize import word_tokenize
from transformers import pipeline # type: ignore
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Initialize the summarization pipeline once
summarizer = pipeline('summarization', model='facebook/bart-large-cnn', min_length=50, max_length=150)

def index(request):
    return render(request, 'index.html')
def split_text_into_chunks(text, max_length):
    # Tokenize the text using NLTK
    words = word_tokenize(text)

    # Initialize variables to store chunks
    chunks = []
    current_chunk = []
    current_length = 0

    # Iterate over words and create chunks
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1  # Adding 1 for the space or punctuation

        if current_length >= max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# to get transcripts 
def get_transcript(request):
    if request.method == 'POST':
        video_id = request.POST.get('video_id')  # Correct way to get POST data
        try:
            transcript =  YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = " ".join([entry['text'] for entry in transcript])
            long_text = transcript_text
            return render(request, 'transcript.html', {'transcript': long_text})
        except Exception as e:
            return render(request, 'transcript.html', {'error': str(e)})
    return render(request, 'index.html')

# to get summary
def get_summary(request):
    if request.method == 'POST':
        video_id = request.POST.get('video_id')  # Correct way to get POST data
        try:
            transcript =  YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = " ".join([entry['text'] for entry in transcript])
            long_text = transcript_text

# Split text into chunks with each size of 2300 tokens
            chunks = split_text_into_chunks(long_text, 2300)

                # Summarize each chunk and concatenate summaries
            summaries = []
            for chunk in chunks:
                summary = summarizer(chunk, max_length=130, do_sample=False)
                summaries.append(summary[0]['summary_text'])

                # Concatenate all summaries into a single text
            final_summary = '.'.join(summaries)
                            

            return render(request, 'summary.html', {'transcript': final_summary})
        except Exception as e:
            return render(request, 'summary.html', {'error': str(e)})
    return render(request, 'index.html')

def get_transcript_and_subtitles(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    transcript_text = " ".join([entry['text'] for entry in transcript])
    
    formatter = TextFormatter()
    subtitle_text = formatter.format_transcript(transcript)
    
    return transcript_text, subtitle_text


def calculate_similarity(text1, text2):
    try:
        stop_words = set(stopwords.words('english'))
        tokens1 = [word for word in word_tokenize(text1.lower()) if word.isalnum() and word not in stop_words]
        tokens2 = [word for word in word_tokenize(text2.lower()) if word.isalnum() and word not in stop_words]

        vectorizer = CountVectorizer().fit_transform([text1, text2])
        vectors = vectorizer.toarray()

        cosine_sim = cosine_similarity(vectors)
        return cosine_sim[0, 1]
    except Exception as e:
        print(f"Error in calculate_similarity: {e}")
        return 0  # Return default value or handle error gracefully


def get_score(request):
    if request.method == 'POST':
        video_id = request.POST.get('video_id')  # Correct way to get POST data
        try:
            transcript_text, subtitle_text = get_transcript_and_subtitles(video_id)
            similarity_score = calculate_similarity(transcript_text, subtitle_text)
            return render(request, 'similarity.html', {'similarity_score': similarity_score})
        except Exception as e:
            return render(request, 'summary.html', {'error': str(e)})
    return render(request, 'index.html')
def get_transcript_long(request):
    if request.method == 'POST':
        video_id = request.POST.get('video_id')

        try:
            transcript_text, subtitle_text = get_transcript_and_subtitles(video_id)
            chunks = split_text_into_chunks(transcript_text, 2300)

            summaries = []
            for chunk in chunks:
                summary = summarizer(chunk, max_length=100, do_sample=False)
                summaries.append(summary[0]['summary_text'])

            final_summary = '. '.join(summaries)
            len_sum = len(summaries)
            t_list = transcript_text.split()

            len_tran = len(t_list)
            similarity_score = calculate_similarity(transcript_text, subtitle_text)
            
            return render(request, 'summary1.html', {
                'transcript': transcript_text,
                'summary': final_summary,
                'similarity_score': similarity_score,
                'len_sum': len_sum,
                'len_tran': len_tran
            })
        except Exception as e:
            return render(request, 'summary1.html', {'error': str(e)})
    return render(request, 'index.html')