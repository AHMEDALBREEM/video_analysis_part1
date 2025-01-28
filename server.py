import warnings
from flask import Flask, request, render_template, send_file, jsonify
import os
import whisper
import tempfile

# Suppress specific warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work")
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")

# Initialize Flask app and configure upload folder
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Whisper model globally
model = whisper.load_model("base")

@app.route('/')
def index():
    """Render the homepage."""
    try:
        return render_template('index.html')  # Ensure 'index.html' exists in 'templates' folder
    except Exception as e:
        return f"Error loading template: {str(e)}", 500




@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads and return the loudest segment."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if not file.filename.lower().endswith(('.mp4', '.mkv', '.avi', '.wav', '.mp3')):
        return jsonify({"error": "Unsupported file format"}), 400

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        file.save(temp_file.name)
        temp_file_path = temp_file.name

    try:
        print("File saved to:", temp_file_path)  # Debug print

        # Process the file to find the loudest segment
        loudest_segment = find_loudest_segment(temp_file_path)
        print("Loudest segment found:", loudest_segment)  # Debug print

        os.remove(temp_file_path)

        return jsonify({"loudest_segment": loudest_segment})

    except Exception as e:
        print(f"Error processing file: {str(e)}")  # Debug print
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

import os
from collections import Counter
import re


def find_loudest_segment(file_path, output_path="segments_output.txt"):
    # Transcribe the audio file using Whisper
    result = model.transcribe(file_path)
    segments = result.get('segments', [])

    # Check if there are any segments
    if not segments:
        raise ValueError("No segments found in the audio.")

    # Prepare the formatted output for all segments
    output_content = "Audio Segments Details:\n========================\n\n"
    
    # Collect all words from the segments for frequency analysis
    all_text = []

    for idx, segment in enumerate(segments):
        segment_text = segment.get("text", "").replace("\n", " ").strip()
        a = len(segment_text.split())
        if a >= 5 : 
            all_text.append(segment_text)

            # Calculate the number of words in the segment
            word_count = len(re.findall(r'\w+', segment_text))

            # Calculate the time duration for the segment
            duration = segment['end'] - segment['start']

            # Calculate words per second for the segment
            words_per_second = word_count / duration if duration > 0 else 0

            output_content += (
                f"Segment {idx + 1}:\n"
                f"Start Time: {segment['start']:.2f} seconds\n"
                f"End Time: {segment['end']:.2f} seconds\n"
                f"Duration: {duration:.2f} seconds\n"
                f"Text: {segment_text}\n"
                f"Word Count: {word_count}\n"
                f"Words per Second: {words_per_second:.2f}\n"
                f"Confidence (avg_logprob): {segment.get('avg_logprob', 'N/A'):.4f}\n"
                f"------------------------\n"
            )

    # Combine all segments into one text block for word frequency analysis
    combined_text = " ".join(all_text)

    # Clean and split the text into words, ignoring common stop words
    words = re.findall(r'\w+', combined_text.lower())
    stop_words = {
        'a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost',
        'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst',
        'amount', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere',
        'are', 'around', 'as', 'at', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming',
        'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between',
        'beyond', 'bill', 'both', 'bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'con',
        'could', 'couldnt', 'cry', 'de', 'describe', 'detail', 'do', 'done', 'down', 'due', 'during',
        'each', 'eg', 'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc', 'even',
        'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen', 'fifty',
        'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found', 'four',
        'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had', 'has', 'hasnt', 'have', 'he',
        'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him',
        'himself', 'his', 'how', 'however', 'hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed',
        'interest', 'into', 'is', 'it', 'its', 'itself', 'keep', 'last', 'latter', 'latterly', 'least',
        'less', 'ltd', 'made', 'many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine', 'more',
        'moreover', 'most', 'mostly', 'move', 'much', 'must', 'my', 'myself', 'name', 'namely',
        'neither', 'never', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor',
        'not', 'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto',
        'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part',
        'per', 'perhaps', 'please', 'put', 'rather', 're', 'same', 'see', 'seem', 'seemed', 'seeming',
        'seems', 'serious', 'several', 'she', 'should', 'show', 'side', 'since', 'sincere', 'six',
        'sixty', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere',
        'still', 'such', 'system', 'take', 'ten', 'than', 'that', 'the', 'their', 'them',
        'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
        'thereupon', 'these', 'they', 'thick', 'thin', 'third', 'this', 'those', 'though', 'three',
        'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward', 'towards',
        'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was',
        'we', 'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter',
        'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
        'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with', 'within',
        'without', 'would', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves'
    }
    filtered_words = [word for word in words if word not in stop_words and len(word) > 4]

    # Count the frequency of each word across all segments
    word_counts = Counter(filtered_words)

    # Get the most common words (highest frequency) and limit to top 10
    most_common_words = word_counts.most_common(10)  # Get top 10 most common words

    # Add the most frequent (loudest) words to the output
    output_content += "\nMost Frequent (Loudest) Words:\n========================\n"
    for word, count in most_common_words:
        output_content += f"Word: {word} - Frequency: {count}\n"

    # Save the formatted output to a text file
    with open(output_path, "w") as f:
        f.write(output_content)

    return output_content




if __name__ == '__main__':
    app.run(debug=True)


