# Sample Data

This directory contains lightweight demonstration assets you can use for local testing and walkthroughs of the clinical note pipeline.

## Contents

- sample_consult.wav / sample_followup.wav – Synthetic mono audio clips for exercising the Whisper transcription endpoint. Pair them with the transcripts below for evaluation.
- sample_consult_transcript.txt / sample_followup_transcript.txt – Mock clinician-patient dialogues that mimic primary-care visits.
- sample_consult_soap.json / sample_followup_soap.json – Reference SOAP notes aligned to each dialogue for ROUGE/BERTScore comparisons.
- generate_samples.py – Utility script that regenerates the demo audio clips if you need fresh artifacts.

> **Note:** Replace these synthetic clips with real de-identified audio (e.g., MedDialog, MTSamples) when demonstrating the full speech-to-note pipeline. Update the reference transcripts and SOAP notes accordingly so evaluation metrics remain meaningful.
