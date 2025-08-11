// index.js
import * as dotenv from 'dotenv';
dotenv.config();

import express from 'express';
import cors from 'cors';
import bodyParser from 'body-parser';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';

const app = express();
app.use(cors({
  origin: 'https://envio-frontend-ytj7.vercel.app',
  methods: ['GET', 'POST'],
  allowedHeaders: ['Content-Type']
}));

app.use(bodyParser.json());

const pinecone = new Pinecone();
const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
let vectorStore;

// Step 1: Index the PDF (only once at startup)
async function indexDocument() {
  const PDF_PATH = './envirotest.pdf';
  const pdfLoader = new PDFLoader(PDF_PATH);
  const rawDocs = await pdfLoader.load();
  console.log("PDF loaded");

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const chunkedDocs = await textSplitter.splitDocuments(rawDocs);
  console.log("Chunking Completed");

  const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
    model: 'text-embedding-004',
  });

  console.log("Embedding model configured");

  vectorStore = await PineconeStore.fromDocuments(chunkedDocs, embeddings, {
    pineconeIndex,
    maxConcurrency: 5,
  });

  console.log("Data Stored successfully");
}

// Step 2: Chat endpoint
app.post('/chat', async (req, res) => {
  try {
    const question = req.body.question;

    if (!question) {
      return res.status(400).json({ error: 'No question provided' });
    }

    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GEMINI_API_KEY,
      model: 'text-embedding-004',
    });

    const results = await vectorStore.similaritySearch(question, 1);
    if (results.length > 0) {
      res.json({ answer: results[0].pageContent });
    } else {
      res.json({ answer: 'I could not find the answer in the provided document.' });
    }
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Something went wrong' });
  }
});

// Step 3: Start the server
app.listen(3000, async () => {
  console.log('Server running on http://localhost:3000');
  await indexDocument(); // Load data when server starts
});
