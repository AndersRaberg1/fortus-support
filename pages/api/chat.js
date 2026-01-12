import { HfInference } from '@huggingface/inference';
import { Pinecone } from '@pinecone-database/pinecone';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { message } = req.body || {};

  try {
    console.log('Chat request:', message);

    const hf = new HfInference(process.env.HUGGINGFACE_API_KEY);
    const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
    const index = pinecone.Index(process.env.PINECONE_INDEX_NAME || 'fortus-support-hf');

    console.log('Embedding message...');
    const embeddingResponse = await hf.featureExtraction({
      model: 'sentence-transformers/all-MiniLM-L6-v2',
      inputs: message,
    });
    const queryEmbedding = Array.from(embeddingResponse);
    console.log('Embedding length:', queryEmbedding.length);

    console.log('Querying Pinecone...');
    const queryResponse = await index.query({
      vector: queryEmbedding,
      topK: 5,
      includeMetadata: true,
    });

    console.log('Matches found:', queryResponse.matches.length);
    const context = queryResponse.matches
      .map(m => m.metadata.text || '')
      .join('\n\n') || 'Ingen relevant kunskap hittades.';

    console.log('Context length:', context.length);

    const prompt = `Du är en hjälpsam support-AI för FortusPay. Använd ENDAST denna kunskap för svaret (inga påhitt): ${context}\n\nFråga: ${message}\nSvara på svenska, kort och stegvis.`;

    console.log('Sending to Groq...');
    const groqResponse = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.GROQ_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'llama-3.1-8b-instant',
        messages: [{ role: 'user', content: prompt }],
      }),
    });

    if (!groqResponse.ok) {
      throw new Error('Groq API fel');
    }

    const data = await groqResponse.json();
    const reply = data.choices[0]?.message?.content?.trim() || 'Inget svar från AI (tom context).';

    console.log('Reply:', reply);
    res.status(200).json({ response: reply });
  } catch (error) {
    console.error('RAG error:', error);
    res.status(500).json({ error: 'Fel vid RAG: ' + (error.message || 'Okänt fel') });
  }
}
