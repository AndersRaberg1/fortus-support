import { NextResponse } from 'next/server';
import Groq from 'groq-sdk';

export async function POST(req) {
  console.log('=== API-anrop startat: POST /api/chat ===');

  try {
    const body = await req.json();
    console.log('Parsad body:', JSON.stringify(body, null, 2));
    const { message } = body;

    // Initiera Groq-klient
    const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

    // Generera AI-svar med Groq
    const completion = await groq.chat.completions.create({
      messages: [
        { role: "system", content: "Du är en hjälpsam support-AI för FortusPay. Svara vänligt på svenska." },
        { role: "user", content: message },
      ],
      model: "mixtral-8x7b-32768", // Eller annan modell från Groq
    });

    const aiReply = completion.choices[0].message.content;
    console.log('AI-svar från Groq:', aiReply);

    console.log('=== API-anrop lyckades ===');
    return NextResponse.json({ reply: aiReply });
  } catch (error) {
    console.error('=== FEL I API ===');
    console.error('Error message:', error.message);
    console.error('Error stack:', error.stack);
    return NextResponse.json({ error: 'Internt serverfel' }, { status: 500 });
  }
}
