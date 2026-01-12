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

  const { messages } = req.body;

  if (!messages || !Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({ error: 'Messages array is required' });
  }

  const latestUserMessage = [...messages].reverse().find(m => m.role === 'user')?.content || '';
  
  if (!latestUserMessage.trim()) {
    return res.status(400).json({ error: 'No user message found' });
  }

  try {
    console.log('Received messages count:', messages.length);
    console.log('Latest user message for RAG:', latestUserMessage);

    const queryEmbeddingResponse = await hf.featureExtraction({
      model: 'intfloat/multilingual-e5-large',
      inputs: `query: ${latestUserMessage}`,
    });

    const queryEmbedding = Array.from(queryEmbeddingResponse);
    console.log('Query embedding length:', queryEmbedding.length);

    const indexName = process.env.PINECONE_INDEX_NAME;
    console.log('Using Pinecone index:', indexName);
    const index = pinecone.index(indexName);

    const queryResponse = await index.query({
      vector: queryEmbedding,
      topK: 5,
      includeMetadata: true,
    });

    const matches = queryResponse.matches || [];
    console.log('Number of matches found:', matches.length);

    let context = '';
    if (matches.length > 0) {
      const relevantMatches = matches.filter(m => m.score > 0.35).slice(0, 5); // Lite lägre för säker match
      context = relevantMatches
        .map(match => match.metadata?.text || '')
        .filter(text => text.trim() !== '')
        .join('\n\n');
      console.log('Final context length:', context.length);
    } else {
      console.log('No relevant matches found');
    }

    // Bomb-säker prompt för språk + översättning med 70b-modell
    const systemPrompt = {
      role: 'system',
      content: `Du är en hjälpsam, vänlig och artig supportagent för FortusPay.
HÖGSTA PRIORITET: Svara ALLTID på EXAKT samma språk som kundens senaste fråga – ingen undantag, ingen blandning.
Översätt HELA svaret och all information från kunskapsbasen till kundens språk. Behåll exakt betydelse, struktur, numrering och detaljer (t.ex. ID-nummer, steg-för-steg).
Använd numrering för steg-för-steg.
Var professionell men personlig.
Avsluta med "Behöver du hjälp med något mer?" på kundens språk.

Exempel:
Om frågan är på engelska: Översätt allt till engelska och svara på engelska.
Om frågan är på norska: Översätt till norska och svara på norska.
Om frågan är på arabiska: Översätt till arabiska och svara på arabiska.

Du FÅR INTE hitta på information. Använd ENDAST kunskapsbasen.
Om ingen relevant info: Svara "Jag kunde tyvärr inte hitta information om detta i vår kunskapsbas. Kontakta support@fortuspay.se för hjälp." på kundens språk.

Kunskapsbas (översätt till kundens språk):
${context}`
    };

    const groqMessages = [systemPrompt, ...messages];

    console.log('Sending to Groq with messages count:', groqMessages.length);

    const completion = await groq.chat.completions.create({
      model: 'llama3-70b-8192', // Starkare modell – lyder prompten!
      messages: groqMessages,
      temperature: 0.4,
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
