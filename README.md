# PDF Content-based Q&A with LangChain and GPT-4

This tool leverages the Retrieval Augmented Generation (RAG) technique, integrating LangChain and GPT-4 to perform content-based question-answering (Q&A) on PDF documents.

## Features

- Upload a PDF document
- Ask questions based on the content of the PDF
- Get answers generated using GPT-4 and LangChain's retrieval capabilities

## Prerequisites

- Python 3.7 or higher
- OpenAI API key

## Installation

1. Clone this repository or copy the code into a local directory.
2. Install the required packages using pip:

    ```bash
    pip install streamlit langchain pymupdf chromadb openai
    ```

## Setup

1. Obtain an OpenAI API key from [OpenAI](https://www.openai.com/).
2. Replace `'VOTRE_CLEF_OPENAI_ici'` in the code with your actual OpenAI API key:

    ```python
    os.environ['OPENAI_API_KEY'] = 'YOUR_OPENAI_API_KEY_HERE'
    ```

## Usage

1. Save the provided code in a file, for example, `demo_RAG.py`.
2. Run the script using the following command:

    ```bash
    streamlit run demo_RAG.py
    ```

3. The Streamlit interface will open in your web browser.
4. Upload the PDF document you want to analyze by clicking on "Upload your PDF".
5. Enter your question in the text input field.
6. The tool will process the PDF and generate an answer based on the content of the PDF and the question asked.

## Code Explanation

- `process_pdf(pdf_path)`: This function loads the PDF document and splits its text into chunks for processing.
- `setup_embeddings_and_vector_db(texts, persist_directory)`: This function sets up embeddings for the text chunks and creates a vector database for efficient retrieval.
- `main()`: This is the main function that sets up the Streamlit interface, handles file uploads, processes the PDF, and generates answers to user questions.

## Example

1. Run the application:

    ```bash
    streamlit run demo_RAG.py
    ```

2. Upload a PDF document.
3. Enter a question such as "What is the main topic of the document?".
4. The answer will be displayed below the input fields.

## Troubleshooting

- Ensure that the PDF file is not corrupted and contains selectable text.
- Check the console for any error messages if the application does not run as expected.

## License

This project is licensed under the MIT License.
