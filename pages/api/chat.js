import { Pinecone } from '@pinecone-database/pinecone';
import { Groq } from 'groq-sdk';
import { HfInference } from '@huggingface/inference';

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const hf = new HfInference(process.env.HF_TOKEN);

const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY,
});

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { message } = req.body;

  if (!message || message.trim() === '') {
    return res.status(400).json({ error: 'Message is required' });
  }

  try {
    console.log('Received message:', message);

    // Generera embedding för frågan med samma modell som vid upload
    console.log('Generating query embedding...');
    const queryEmbeddingResponse = await hf.featureExtraction({
      model: 'sentence-transformers/all-MiniLM-L6-v2', // 384 dim, samma som vid upload
      inputs: message,
    });

    // Konvertera till float array
    const queryEmbedding = Array.from(queryEmbeddingResponse);

    console.log('Query embedding length:', queryEmbedding.length); // ska vara 384

    // Hämta Pinecone index
    const indexName = process.env.PINECONE_INDEX_NAME;
    console.log('Using Pinecone index:', indexName);

    const index = pinecone.index(indexName);

    // Query Pinecone
    const queryResponse = await index.query({
      vector: queryEmbedding,
      topK: 10, // Ökat temporärt för debug
      includeMetadata: true,
      // includeValues: true, // Uncomment om du vill se vektorer
    });

    const matches = queryResponse.matches || [];
    console.log('Raw Pinecone response:', JSON.stringify(queryResponse, null, 2));
    console.log('Number of matches found:', matches.length);

    let context = '';
    if (matches.length > 0) {
      // Ta de 5 bästa (eller alla om färre)
      const topMatches = matches.slice(0, 5);
      context = topMatches
        .map((match, i) => {
          console.log(`Match ${i + 1} score:`, match.score);
          console.log(`Match ${i + 1} metadata preview:`, match.metadata?.text?.substring(0, 200));
          return match.metadata?.text || '';
        })
        .filter(text => text.trim() !== '')
        .join('\n\n');

      console.log('Final context length sent to Groq:', context.length);
    } else {
      console.log('No matches returned – possible causes: index name, embedding mismatch, or empty index');
      console.log('Final context length sent to Groq: 0 (no context)');
    }

    // Bygg prompt till Groq
    const systemPrompt = `Du är en hjälpsam och vänlig supportagent för FortusPay. 
Använd ENDAST information från följande kunskapsbas för att svara. 
Om frågan inte kan besvaras med denna information, säg "Jag kunde tyvärr inte hitta information om detta i vår kunskapsbas. Kontakta support@fortuspay.se för hjälp."

Kunskapsbas:
${context}

Svara kort, tydligt och på svenska.`;

    console.log('Sending to Groq...');

    const completion = await groq.chat.completions.create({
      model: 'llama-3.1-8b-instant',
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: message },
      ],
      temperature: 0.3,
      max_tokens: 1024,
      stream: false,
    });

    const answer = completion.choices[0]?.message?.content || 'Inget svar från modellen.';

    console.log('Groq response:', answer.substring(0, 500));

    res.status(200).json({ answer });
  } catch (error) {
    console.error('Error in chat API:', error);
    res.status(500).json({ error: 'Internal server error', details: error.message });
  }
}

export const config = {
  api: {
    bodyParser: true,
  },
};
