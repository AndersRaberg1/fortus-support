import { Pinecone } from '@pinecone-database/pinecone';
import { HfInference } from '@huggingface/inference';
import { v4 as uuidv4 } from 'uuid';

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const hf = new HfInference(process.env.HF_TOKEN);

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed – use POST' });
  }

  const { chunks } = req.body; // Förväntar array av { keyword: string, text: string }

  if (!chunks || !Array.isArray(chunks) || chunks.length === 0) {
    return res.status(400).json({ error: 'Body måste innehålla "chunks" array med objekt { keyword, text }' });
  }

  try {
    const indexName = process.env.PINECONE_INDEX_NAME;
    if (!indexName) {
      return res.status(500).json({ error: 'PINECONE_INDEX_NAME saknas i env' });
    }

    console.log('Using Pinecone index:', indexName);
    const index = pinecone.index(indexName);

    const vectors = [];

    for (const chunk of chunks) {
      if (!chunk.text || chunk.text.trim() === '') {
        console.warn('Hoppar över tom chunk:', chunk.keyword);
        continue;
      }

      console.log(`Genererar embedding för "${chunk.keyword}"...`);

      // Viktigt: "passage: " prefix för dokument (e5-modell)
      const embeddingResponse = await hf.featureExtraction({
        model: 'intfloat/multilingual-e5-large',
        inputs: `passage: ${chunk.text.trim()}`,
      });

      const embedding = Array.from(embeddingResponse);

      if (embedding.length !== 1024) {
        console.warn(`Ovänat embedding längd för ${chunk.keyword}: ${embedding.length}`);
      }

      const id = uuidv4(); // Unikt ID – eller använd keyword som bas om du vill

      vectors.push({
        id,
        values: embedding,
        metadata: {
          keyword: chunk.keyword || 'Okänd',
          text: chunk.text.trim(),
        },
      });
    }

    if (vectors.length === 0) {
      return res.status(400).json({ error: 'Inga giltiga chunks att ladda upp' });
    }

    // Upsert i batch (Pinecone hanterar stora batcher bra)
    console.log(`Upsertar ${vectors.length} vectors till Pinecone...`);
    await index.upsert(vectors);

    console.log(`Uppladdat ${vectors.length} chunks framgångsrikt!`);

    res.status(200).json({
      success: true,
      message: `Uppladdat ${vectors.length} chunks till index "${indexName}"`,
      uploadedCount: vectors.length,
    });
  } catch (error) {
    console.error('Fel vid upload till Pinecone:', error);
    res.status(500).json({
      error: 'Internt fel vid upload',
      details: error.message,
    });
  }
}

export const config = {
  api: {
    bodyParser: true,
  },
};
