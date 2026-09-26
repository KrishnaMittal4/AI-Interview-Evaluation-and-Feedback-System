// src/api.js - Interview Analyzer API Client

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function analyzeText({ answer, question, questionType }) {
  const form = new FormData();
  form.append("answer", answer);
  form.append("question", question);
  form.append("question_type", questionType);

  const res = await fetch(`${BASE_URL}/analyze/text`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Analysis failed");
  }

  return res.json();
}

export async function analyzeAudio({ audioBlob, question, questionType }) {
  const form = new FormData();
  form.append("audio", audioBlob, "recording.webm");
  form.append("question", question);
  form.append("question_type", questionType);

  const res = await fetch(`${BASE_URL}/analyze/audio`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Audio analysis failed");
  }

  return res.json();
}

export async function getQuestions() {
  const res = await fetch(`${BASE_URL}/questions`);
  return res.json();
}

export async function checkHealth() {
  const res = await fetch(`${BASE_URL}/health`);
  return res.json();
}
