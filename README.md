Clinical Note Pipeline

An end-to-end project that turns clinician–patient conversations into structured SOAP notes using Whisper for transcription, transformer models for summarization, and modern deployment tools.

Features

Real-time transcription: FastAPI WebSocket streams Whisper outputs for low-latency updates.

Structured summaries: Hugging Face models generate SOAP-formatted notes.

Metrics & logging: ROUGE and BERTScore stored in PostgreSQL for each note.

Deployment-ready: Includes Docker image, Kubernetes manifests, and Azure ML deployment templates.

Web dashboard: Next.js frontend for audio upload, live transcripts, and summaries.

Quick Start
Backend
cd backend
pip install -r requirements.txt
uvicorn app:app --reload

Frontend
cd ../frontend/next-dashboard
npm install && npm run dev


Visit http://localhost:3000
 to upload audio and generate SOAP notes.

Fine-Tuning

Fine-tune the summarizer (default: facebook/bart-large-cnn) on de-identified datasets like MIMIC-III or MedDialog, then update SUMMARIZER_MODEL_NAME to your trained checkpoint.

Deployment

Docker: docker build -t clinical-pipeline .

Kubernetes: apply manifests in /k8s

Azure ML: optional GPU/CPU scaling configs included