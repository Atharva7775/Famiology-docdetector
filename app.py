# from dotenv import load_dotenv
import streamlit as st
from PIL import Image
# import fitz 
# import pdf2image
import io
from transformers import CLIPProcessor, CLIPModel
# from pdf2image import convert_from_bytes
from PyPDF2 import PdfFileReader
CLIPProcessor.safety_checker = None
# CLIPProcessor.safety_checker = None
# CLIPModel.safety_checker = None
# Adjust img2text to handle both file-like objects and PIL.Image.Image instances
def img2text(uploaded_file):
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    if isinstance(uploaded_file, Image.Image):
        image = uploaded_file
    else:
        image = Image.open(uploaded_file)

    array = ["Passport", "Driver License", "Green Card", "401K-statement", "Last-will-and-testament", "life-insurance", "W2-form", "f8889_HSA", "other"]
    inputs = processor(text=array, images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  
    probs = logits_per_image.softmax(dim=1)  

    probs = probs.tolist()
    flat_probs = [prob for sublist in probs for prob in sublist]
    max_prob = max(flat_probs)
    index_of_max = flat_probs.index(max_prob)
    
    st.write("Your documents have been uploaded successfully. Thanks for submitting your ", array[index_of_max], ".")
    st.write("We'll take care of the rest.")
    # st.write("Accuracy - ", max_prob)
    # return array[index_of_max]


# st.sidebar.info("Hello World")
def main():
    # img = Image.open('/Users/atharvabapat/Desktop/Theoremlabs-project/favicon (2).ico')
    # st.set_page_config(page_title="Document Identification")
    
    st.set_page_config(page_title='Famiology.docdetector', page_icon='/Users/atharvabapat/Desktop/Theoremlabs-project/favicon (2).ico')
    st.header('Famiology Document Detector')
    st.sidebar.image("FamiologyTextLogo.png", use_column_width=True)
    
    with st.sidebar:
        st.header('About App')
        st.header('Smart Document Recognition: Instantly Identify Uploaded Documents')
        st.sidebar.info('Empower your document management process with Smart Document Recognition. This advanced feature swiftly identifies the type of document you upload, making document handling effortless and efficient.')
        st.header('How It Works: ')
        expander = st.expander("See Details")
        expander.write('<ins>Upload Your Document:</ins> Select the document you wish to process using the provided file upload button. \n\n Intelligent Analysis: Our system employs cutting-edge technology to analyze the documents structure, layout, and content. \n\n Automatic Identification: Based on the analysis, Smart Document Recognition accurately identifies the document type, whether its an identification document, real estate document, 401k document or any other document format. \n\n Streamlined Processing: With the document type identified, our platform can seamlessly route it to the appropriate workflow or apply predefined actions, saving you valuable time and effort.')
        st.header('What Problem it Solves?')
        expander = st.expander("See Details")
        expander.write('Efficiency: Instantly recognize document types without manual intervention. \n\n Accuracy: Ensure accurate processing and categorization of documents. \n\n Productivity: Automate document handling workflows for smoother operations.')
        st.header('Value') 
        expander = st.expander("See Details")
        expander.write('eVaults are smart and can support automation of client interactions as well as parallel internal ops process . Saves ops time, cleaner data, nudges for clients as well as for internal staff.')
    uploaded_file = st.file_uploader("Choose a file to upload", type=['png', 'jpeg', 'jpg', 'pdf'])
    
    if uploaded_file is not None:
        # Display the uploaded image
        if uploaded_file.type == 'application/pdf':
            uploaded_file = pdf_to_img(uploaded_file)
            # st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
            # scenario1 = img2text(uploaded_file)
            img2text(uploaded_file)
            # with st.expander("Identified Document Type"):
            #     print("Thank You for uploading ", st.write(scenario1))
        else:    
            # st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        
            # scenario = img2text(uploaded_file) 
            img2text(uploaded_file)
        
            # with st.expander("Extracted Text"):
            #     print("Thank You for uploading ", st.write(scenario))

def pdf_to_img(uploaded_file):
    # Read the PDF file
    pdf_data = uploaded_file.read()

    # Open the PDF using PyPDF2
    pdf_reader = PdfFileReader(io.BytesIO(pdf_data))

    # Check if the PDF has any pages
    if pdf_reader.numPages == 0:
        raise ValueError("The PDF file is empty.")

    # Extract the first page of the PDF
    first_page = pdf_reader.getPage(0)

    # Convert the PDF page to an image using PIL
    image = first_page.to_pil()

    return image


if __name__ == '__main__':
        main()
