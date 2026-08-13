# Custom ASR: Streaming-Optionen (Rangliste)

Kontext: Der Custom-Server-Provider (`OpenAICompatibleTranscriptionProvider`)
schickt pro Äusserung einen HTTP-Request an einen OpenAI-kompatiblen
`/audio/transcriptions`-Endpoint (oMLX mit
`gcoli/whisper-large-v3-swiss-german-mlx-fp16`). Ein HTTP-Roundtrip liefert
keine Live-Vorschau während des Sprechens. Dieses Dokument hält die geprüften
Ansätze fest, wie eine Vorschau trotzdem möglich ist — inklusive des gewählten.

Gemessene Referenz (2026-08-13, M-Series, oMLX Q8): 12.4 s Audio in 783 ms
transkribiert, RTF 0.063.

## Rangliste

### 1. Zwei-Modell-Ansatz — GEWÄHLT UND IMPLEMENTIERT

Vorschau und Finale sind getrennte Engines; das `TranscriptionProvider`-Protokoll
sieht das explizit vor (`transcribeStreaming` = "faster/lighter paths",
`transcribeFinal` = Qualität).

- **Vorschau:** `gcoli/whisper-large-v3-swiss-german-gguf-q8_0` (1.67 GB,
  gebaut für `handy-computer/transcribe.cpp`), läuft **in-process** über den
  bestehenden `WhisperProvider` — kein Netzwerk, konstante Serverlast null.
  In der App als eigenständiges Modell "Whisper Swiss German (Q8)" wählbar
  (damit auch komplett offline diktierbar).
- **Finale:** fp16 via oMLX (maximale Genauigkeit, ein Request pro Äusserung).
- Graceful Degradation: Ist das Q8-Modell nicht installiert, zeigt das Overlay
  wie zuvor nur die Wellenform; das Diktat funktioniert unverändert.

### 2. Sliding Window mit Prefix-Commit (LocalAgreement)

Aufeinanderfolgende Hypothesen vergleichen, den stabilen Präfix committen und
aus dem Audiopuffer abschneiden (bekannt aus `whisper_streaming`/WhisperLive).
Puffer bleibt beschränkt, Vorschau flackert kaum. **Upgrade-Pfad**, falls
Ein-Modell-Betrieb über den Server gewünscht ist; Logik lebt komplett im
Provider (`transcribeStreaming`), kein Server-Umbau.

### 3. VAD-Segmentierung mit Prompt-Verkettung

Sprechpausen per Voice-Activity-Detection erkennen, jedes Segment genau einmal
transkribieren (O(n) statt O(n²)), Kontextverlust über das `prompt`-Feld des
Transcriptions-Endpoints mildern (Text des Vorsegments mitgeben).
Ressourcenschonendste Variante; Text erscheint satzweise bei Pausen statt
wortweise.

### 4. Nativ streamende Architektur über WebSocket — GEPARKT

Modell mit kausaler Attention, das Tokens inkrementell emittiert, während Audio
hineinfliesst. Braucht einen Streaming-Transport (WebSocket/Realtime-API)
zwischen App und Server — grösserer Fork (neuer Transportpfad statt
`URLSession.data`). Ob oMLX einen solchen exponiert: unverifiziert.

#### Natives Streaming heute schon — unter Verzicht auf oMLX

Wer echtes natives Streaming jetzt will, hat zwei Wege, beide ohne oMLX:

1. **Eingebaute Streaming-Engines von FluidVoice:** Parakeet Flash und
   Nemotron 3.5 Streaming laufen in-process mit echtem inkrementellem Decoding
   (kein Chunk-Polling). Einschränkung: kein Schweizerdeutsch.
2. **Voxtral-Mini-4B-Realtime-2602** (Mistral, Apache 2.0, auf
   Ministral-3-3B-Base): nativ streamende ASR-Architektur, konfigurierbares
   Delay 80 ms–2.4 s, ~8.7 % WER multilingual bei 480 ms, 13 Sprachen inkl.
   Standarddeutsch. Empfohlene Runtime laut Model Card ist **vLLM**; daneben
   existieren Transformers-, ExecuTorch- und MLX-Community-Implementierungen.
   Anbindung erfordert einen eigenen WebSocket-/Realtime-Transportpfad im
   Provider. Kein Schweizerdeutsch ohne eigenes Fine-tune — als
   Streaming-Komplement zur Whisper-Familie gedacht, nicht als Ersatz.

Kombinierbar: Ansatz 1 (Zwei-Modell) plus Ansatz 2 als Commit-Logik über der
Vorschau wäre die Ausbaustufe, falls die Vorschau bei langen Diktaten zu träge
wird (lokale large-Q8-Pässe wachsen mit der Pufferlänge).
