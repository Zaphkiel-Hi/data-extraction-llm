

!mkdir pdfs
     

!gdown https://drive.google.com/uc?id=1DnG_6LoXjn57oGGP5jfLvTxCRoRy87qz -O pdfs/Insurance.pdf

     




     

!rm -rf "db"
     

loader = PyPDFDirectoryLoader("pdfs")
docs = loader.load()
len(docs)
     

embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large", model_kwargs={"device": DEVICE}
)
     

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
texts = text_splitter.split_documents(docs)
len(texts)
     

%%time
db = Chroma.from_documents(texts, embeddings, persist_directory="db")
     

