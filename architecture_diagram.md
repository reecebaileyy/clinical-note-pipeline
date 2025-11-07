# Architecture Diagram

```mermaid
graph LR
    A[Audio Input (.wav/.mp3)] -->|WebSocket / REST| B[Whisper Transcription Service]
    B -->|Partial transcript stream| C[FastAPI Orchestrator]
    C -->|Transcript text| D[Summarization Model API]
    D -->|SOAP sections| E[SOAP Note Formatter]
    E -->|Structured note + summary| F[Evaluation Logger]
    F -->|ROUGE / BERTScore + transcripts| G[(PostgreSQL)]
    E -->|SOAP note + metrics| H[Next.js Dashboard]
    B -->|Streamed transcript updates| H
    G -->|Historical insights| H
```

```mermaid
sequenceDiagram
    participant Client
    participant Backend as FastAPI Backend
    participant Whisper as Whisper Service
    participant Summarizer as Summarizer Service
    participant DB as PostgreSQL

    Client->>Backend: Upload audio chunk (WebSocket /transcribe)
    Backend->>Whisper: Append chunk & transcribe
    Whisper-->>Backend: Partial transcript + language
    Backend-->>Client: Stream partial transcript JSON
    Client->>Backend: Finalize session
    Backend->>Summarizer: POST transcript for SOAP summary
    Summarizer-->>Backend: SOAP note + raw summary
    Backend->>DB: Persist transcript, SOAP note, metrics
    Backend-->>Client: Return SOAP note + evaluation metrics
```
