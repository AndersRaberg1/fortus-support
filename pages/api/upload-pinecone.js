import { Pinecone } from '@pinecone-database/pinecone';
import { v4 as uuid } from 'uuid';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
    const index = pinecone.Index('fortus-support');

    const chunks = [ /* samma chunks som tidigare, kopiera från mitt förra svar */ ];

    const vectors = [];
    for (const chunk of chunks) {
      const embedResponse = await pinecone.inference.embed(
        'llama-text-embed-v2',
        [chunk.text],
        { input_type: 'passage' }  // Krävs för chunks!
      );
      const embedding = embedResponse.data[0].values;
      vectors.push({
        id: uuid(),
        values: embedding,
        metadata: { keyword: chunk.keyword, text: chunk.text }
      });
    }

    await index.upsert(vectors);
    res.status(200).json({ message: `Uppladdat ${vectors.length} chunks framgångsrikt!` });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: error.message });
  }
}
