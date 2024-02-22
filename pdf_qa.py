import os
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import HuggingFacePipeline
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util

import re

# Constants
EMB_OPENAI_ADA = "text-embedding-ada-002"
EMB_INSTRUCTOR_XL = "hkunlp/instructor-xl"
EMB_SBERT_MPNET_BASE = "sentence-transformers/all-mpnet-base-v2" # Chroma takes care if embeddings are None
EMB_SBERT_MINILM = "sentence-transformers/all-MiniLM-L6-v2" # Chroma takes care if embeddings are None


LLM_OPENAI_GPT35 = "gpt-3.5-turbo"
LLM_FLAN_T5_XXL = "google/flan-t5-xxl"
LLM_FLAN_T5_XL = "google/flan-t5-xl"
LLM_FASTCHAT_T5_XL = "lmsys/fastchat-t5-3b-v1.0"
LLM_GO_BRUINS = "rwitz/go-bruins-v2"
LLM_FLAN_T5_SMALL = "google/flan-t5-small"
LLM_FLAN_T5_BASE = "google/flan-t5-base"
LLM_FLAN_T5_LARGE = "google/flan-t5-large"
LLM_FALCON_SMALL = "tiiuae/falcon-7b-instruct"

os.environ['OPENAI_API_KEY']  = "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"
class PdfQA:
    def __init__(self,config:dict = {}):
        self.config = config
        self.embedding = None
        self.vectordb = None
        self.llm = None
        self.qa = None
        self.retriever = None

    # The following class methods are useful to create global GPU model instances
    # This way we don't need to reload models in an interactive app,
    # and the same model instance can be used across multiple user sessions
    @classmethod
    def create_instructor_xl(cls):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HuggingFaceInstructEmbeddings(model_name=EMB_INSTRUCTOR_XL, model_kwargs={"device": device})

    @classmethod
    def create_open_ai_emb(cls):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return OpenAIEmbeddings()

    @classmethod
    def create_sbert_mpnet(cls):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HuggingFaceEmbeddings(model_name=EMB_SBERT_MPNET_BASE, model_kwargs={"device": device})
    
    @classmethod
    def create_flan_t5_small(cls, load_in_8bit=False):
        # Local flan-t5-small for inference
        # Wrap it in HF pipeline for use with LangChain
        model="google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model)
        return pipeline(
            task="text2text-generation",
            model=model,
            tokenizer = tokenizer,
            max_new_tokens=100,
            model_kwargs={"device_map": "auto", "load_in_8bit": load_in_8bit, "max_length": 512, "temperature": 0.}
        )
    @classmethod
    def create_flan_t5_base(cls, load_in_8bit=False):
        # Wrap it in HF pipeline for use with LangChain
        model="google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model)
        return pipeline(
            task="text2text-generation",
            model=model,
            tokenizer = tokenizer,
            max_new_tokens=100,
            model_kwargs={"device_map": "auto", "load_in_8bit": load_in_8bit, "max_length": 512, "temperature": 0.}
        )
    @classmethod
    def create_flan_t5_large(cls, load_in_8bit=False):
        # Wrap it in HF pipeline for use with LangChain
        model="google/flan-t5-large"
        tokenizer = AutoTokenizer.from_pretrained(model)
        return pipeline(
            task="text2text-generation",
            model=model,
            tokenizer = tokenizer,
            max_new_tokens=100,
            model_kwargs={"device_map": "auto", "load_in_8bit": load_in_8bit, "max_length": 512, "temperature": 0.}
        )
    
    def init_embeddings(self) -> None:
        # OpenAI ada embeddings API
        if self.config["embedding"] == EMB_OPENAI_ADA:
            self.embedding = OpenAIEmbeddings()
        elif self.config["embedding"] == EMB_INSTRUCTOR_XL:
            # Local INSTRUCTOR-XL embeddings
            if self.embedding is None:
                self.embedding = PdfQA.create_instructor_xl()
        elif self.config["embedding"] == EMB_SBERT_MPNET_BASE:
            ## this is for SBERT
            if self.embedding is None:
                self.embedding = PdfQA.create_sbert_mpnet()
        else:
            self.embedding = None ## DuckDb uses sbert embeddings
            # raise ValueError("Invalid config")

    def init_models(self) -> None:
        """ Initialize LLM models based on config """
        load_in_8bit = self.config.get("load_in_8bit",False)
        
        print(self.config["llm"])
        # OpenAI GPT 3.5 API
        if self.config["llm"] == LLM_OPENAI_GPT35:
            # OpenAI GPT 3.5 API
            pass
        elif self.config["llm"] == LLM_FLAN_T5_SMALL:
            if self.llm is None:
                self.llm = PdfQA.create_flan_t5_small(load_in_8bit=load_in_8bit)
        elif self.config["llm"] == LLM_FLAN_T5_BASE:
            if self.llm is None:
                self.llm = PdfQA.create_flan_t5_base(load_in_8bit=load_in_8bit)
        elif self.config["llm"] == LLM_FLAN_T5_LARGE:
            if self.llm is None:
                self.llm = PdfQA.create_flan_t5_large(load_in_8bit=load_in_8bit)
        elif self.config["llm"] == LLM_FLAN_T5_XL:
            if self.llm is None:
                self.llm = PdfQA.create_flan_t5_xl(load_in_8bit=load_in_8bit)
        elif self.config["llm"] == LLM_FLAN_T5_XXL:
            if self.llm is None:
                self.llm = PdfQA.create_flan_t5_xxl(load_in_8bit=load_in_8bit)
        elif self.config["llm"] == LLM_GO_BRUINS:
            if self.llm is None:
                self.llm = PdfQA.create_go_bruins(load_in_8bit=load_in_8bit)
        elif self.config["llm"] == LLM_FALCON_SMALL:
            if self.llm is None:
                self.llm = PdfQA.create_falcon_instruct_small(load_in_8bit=load_in_8bit)
        else:
            raise ValueError("Invalid config")        
    def vector_db_pdf(self) -> None:
        """
        creates vector db for the embeddings and persists them or loads a vector db from the persist directory
        """
        pdf_path = self.config.get("pdf_path",None)
        persist_directory = self.config.get("persist_directory",None)
        if persist_directory and os.path.exists(persist_directory):
            ## Load from the persist db
            self.vectordb = Chroma(persist_directory=persist_directory, embedding_function=self.embedding)
        elif pdf_path and os.path.exists(pdf_path):
            ## 1. Extract the documents
            loader = PDFPlumberLoader(pdf_path)
            documents = loader.load()
            ## 2. Split the texts
            text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
            texts = text_splitter.split_documents(documents)
            # text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=10, encoding_name="cl100k_base")  # This the encoding for text-embedding-ada-002
            text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=10)  # This the encoding for text-embedding-ada-002
            texts = text_splitter.split_documents(texts)

            ## 3. Create Embeddings and add to chroma store
            ##TODO: Validate if self.embedding is not None
            self.vectordb = Chroma.from_documents(documents=texts, embedding=self.embedding, persist_directory=persist_directory)
        else:
            raise ValueError("NO PDF found")

    def retreival_qa_chain(self):
        """
        Creates retrieval qa chain using vectordb as retrivar and LLM to complete the prompt
        """
        ##TODO: Use custom prompt
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k":3})
        
        if self.config["llm"] == LLM_OPENAI_GPT35:
          # Use ChatGPT API
          self.qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=self.vectordb.as_retriever(search_kwargs={"k":3}))
        else:
            hf_llm = HuggingFacePipeline(pipeline=self.llm,model_id=self.config["llm"])

            self.qa = RetrievalQA.from_chain_type(llm=hf_llm, chain_type="stuff",retriever=self.retriever)
            if self.config["llm"] == LLM_FLAN_T5_SMALL or self.config["llm"] == LLM_FLAN_T5_BASE or self.config["llm"] == LLM_FLAN_T5_LARGE:
                question_t5_template = """
                context: {context}
                question: {question}
                answer: 
                """
                QUESTION_T5_PROMPT = PromptTemplate(
                    template=question_t5_template, input_variables=["context", "question"]
                )
                self.qa.combine_documents_chain.llm_chain.prompt = QUESTION_T5_PROMPT
            self.qa.combine_documents_chain.verbose = True
            self.qa.return_source_documents = True
    def answer_query(self,query:str) ->str:
        """
        Answer the question
        """
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        def semantic_similarity(query, context,threshold=0.2):
            query_embedding = model.encode(query, convert_to_tensor=True)
            context_embedding = model.encode(context, convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(query_embedding, context_embedding)
            print(cosine_scores)
            return cosine_scores.item()>=threshold

        
        retrieved_docs = self.retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        
            # Check if the query is relevant to the context
        if semantic_similarity(query, context):
            # Proceed with generating an answer using the language model
            response = self.qa({"query": query}, return_only_outputs=False)
            return response ['result']
        else:
            # Return a response indicating the lack of relevant context
            return "The answer is not in context"