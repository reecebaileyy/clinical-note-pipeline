'use client';

import { FormEvent, useCallback, useMemo, useRef, useState } from 'react';

import { MetricsTable } from '../components/MetricsTable';
import { SoapNoteCard } from '../components/SoapNoteCard';

type SoapNote = {
  subjective: string;
  objective: string;
  assessment: string;
  plan: string;
};

type SummarizeResponse = {
  soap_note: SoapNote;
  raw_summary: string;
  metrics?: Record<string, number>;
  session_id?: string | null;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000';

function resolveWebsocketUrl(baseUrl: string): string {
  if (baseUrl.startsWith('https://')) {
    return baseUrl.replace('https://', 'wss://');
  }
  if (baseUrl.startsWith('http://')) {
    return baseUrl.replace('http://', 'ws://');
  }
  return `ws://${baseUrl.replace(/^ws(s)?:\/\//, '')}`;
}

export default function DashboardPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [transcript, setTranscript] = useState<string>('');
  const [soapNote, setSoapNote] = useState<SoapNote | null>(null);
  const [metrics, setMetrics] = useState<Record<string, number> | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string>('');
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const websocketRef = useRef<WebSocket | null>(null);

  const handleFileChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    setSelectedFile(file ?? null);
  }, []);

  const uploadTranscription = useCallback(async () => {
    if (!selectedFile) {
      setStatusMessage('Please choose an audio file first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    setStatusMessage('Uploading audio for batch transcription...');
    const response = await fetch(`${API_BASE}/transcribe`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      setStatusMessage('Transcription failed. Please check server logs.');
      return;
    }

    const payload = await response.json();
    setTranscript(payload.text);
    setSessionId(payload.session_id ?? null);
    setStatusMessage('Batch transcription completed.');
  }, [selectedFile]);

  const summarizeTranscript = useCallback(
    async (event?: FormEvent) => {
      event?.preventDefault();
      if (!transcript.trim()) {
        setStatusMessage('Nothing to summarize yet. Add or stream a transcript first.');
        return;
      }

      setStatusMessage('Generating SOAP summary...');
      const response = await fetch(`${API_BASE}/summarize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transcript, session_id: sessionId }),
      });

      if (!response.ok) {
        setStatusMessage('Summarization failed.');
        return;
      }

      const payload: SummarizeResponse = await response.json();
      setSoapNote(payload.soap_note);
      setMetrics(payload.metrics ?? null);
      setStatusMessage('SOAP note generated successfully.');
    },
    [sessionId, transcript],
  );

  const stopStreaming = useCallback(() => {
    websocketRef.current?.close(1000, 'client-stop');
    websocketRef.current = null;
    setIsStreaming(false);
    setStatusMessage('Streaming stopped. You can finalize summary now.');
  }, []);

  const streamTranscription = useCallback(async () => {
    if (!selectedFile) {
      setStatusMessage('Select an audio file to stream.');
      return;
    }

    const wsUrl = `${resolveWebsocketUrl(API_BASE)}/ws/transcribe`;
    const websocket = new WebSocket(wsUrl);
    websocketRef.current = websocket;
    setIsStreaming(true);
    setStatusMessage('Connecting to streaming service...');

    websocket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      switch (message.event) {
        case 'session_started':
          setSessionId(message.session_id);
          break;
        case 'partial_transcript':
          setTranscript(message.text ?? '');
          break;
        case 'config_applied':
          setStatusMessage(`Language configured: ${message.language ?? 'auto'}`);
          break;
        case 'final_transcript':
          setTranscript(message.text ?? '');
          setStatusMessage('Streaming transcription finalized.');
          setIsStreaming(false);
          websocket.close();
          break;
        case 'error':
          setStatusMessage(`Streaming error: ${message.detail}`);
          setIsStreaming(false);
          websocket.close();
          break;
        default:
          break;
      }
    };

    websocket.onerror = () => {
      setStatusMessage('WebSocket error encountered.');
      setIsStreaming(false);
    };

    websocket.onclose = () => {
      setIsStreaming(false);
    };

    const chunkSize = 32 * 1024;
    const file = selectedFile;
    let offset = 0;

    const sendNextChunk = async () => {
      if (!websocketRef.current || websocketRef.current.readyState !== WebSocket.OPEN) {
        return;
      }

      const slice = file.slice(offset, offset + chunkSize);
      const buffer = await slice.arrayBuffer();
      websocket.send(buffer);
      offset += chunkSize;

      if (offset < file.size) {
        setTimeout(sendNextChunk, 75);
      } else {
        websocket.send(JSON.stringify({ event: 'finalize' }));
      }
    };

    websocket.onopen = () => {
      setStatusMessage('Streaming connection established. Sending audio...');
      sendNextChunk();
    };
  }, [selectedFile]);

  const websocketControls = useMemo(() => {
    if (isStreaming) {
      return (
        <button style={{ background: '#dc2626', color: '#fff' }} onClick={stopStreaming}>
          Stop Streaming
        </button>
      );
    }
    return (
      <button style={{ background: '#2563eb', color: '#fff' }} onClick={streamTranscription}>
        Start Streaming
      </button>
    );
  }, [isStreaming, stopStreaming, streamTranscription]);

  return (
    <main>
      <h1>Clinical Note Pipeline Dashboard</h1>
      <p style={{ maxWidth: '720px', marginBottom: '2rem', color: '#4b5563' }}>
        Upload clinical conversation audio, watch Whisper-driven transcriptions stream in real time, and
        convert transcripts into structured SOAP notes for audit-ready documentation.
      </p>

      <div className="grid two" style={{ marginBottom: '1.5rem' }}>
        <section>
          <h2 style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>Audio Controls</h2>
          <input type="file" accept="audio/wav,audio/mp3,audio/mpeg" onChange={handleFileChange} />
          <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1rem', flexWrap: 'wrap' }}>
            <button style={{ background: '#10b981', color: '#fff' }} onClick={uploadTranscription} disabled={!selectedFile}>
              Batch Transcribe
            </button>
            {websocketControls}
          </div>
          {statusMessage && <div className="status-banner">{statusMessage}</div>}
        </section>

        <section>
          <h2 style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>Transcript</h2>
          <textarea
            value={transcript}
            onChange={(event) => setTranscript(event.target.value)}
            placeholder="Streaming transcript will appear here."
          />
          <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '0.75rem' }}>
            <button style={{ background: '#9333ea', color: '#fff' }} onClick={(event) => summarizeTranscript(event)}>
              Summarize to SOAP
            </button>
          </div>
        </section>
      </div>

      <div className="grid two">
        <SoapNoteCard note={soapNote ?? { subjective: '', objective: '', assessment: '', plan: '' }} />
        <MetricsTable metrics={metrics} />
      </div>
    </main>
  );
}


