import Foundation

// MARK: - Custom OpenAI-compatible Transcription Provider

/// Errors surfaced to the user when talking to the remote transcription server.
enum OpenAICompatibleTranscriptionError: LocalizedError {
    case invalidBaseURL(String)
    case missingModelName
    case emptyAudio
    case serverError(statusCode: Int, message: String?)
    case invalidResponse

    var errorDescription: String? {
        switch self {
        case .invalidBaseURL(let url):
            return "Custom ASR server URL is invalid: \(url)"
        case .missingModelName:
            return "No model name configured for the custom ASR server."
        case .emptyAudio:
            return "No audio captured to transcribe."
        case let .serverError(statusCode, message):
            if statusCode == 401 {
                return "Custom ASR server rejected the API key (HTTP 401). Check the key in Voice Engine settings."
            }
            let detail = message.map { ": \($0)" } ?? ""
            return "Custom ASR server returned HTTP \(statusCode)\(detail)"
        case .invalidResponse:
            return "Custom ASR server returned an unreadable response."
        }
    }
}

/// A TranscriptionProvider that posts recorded audio to any OpenAI-compatible
/// `/audio/transcriptions` endpoint (e.g. a local oMLX server hosting a custom
/// Whisper model). No local model download; configuration lives in SettingsStore.
final class OpenAICompatibleTranscriptionProvider: TranscriptionProvider {
    var name: String { "Custom Server (OpenAI API)" }

    var isAvailable: Bool { true }

    private(set) var isReady: Bool = false

    var shouldClearCacheAfterCancellation: Bool { false }

    private let session: URLSession

    /// Local Swiss German Q4 GGUF used for live streaming previews only.
    /// The final transcription always goes to the remote server; if the local
    /// model is not installed, previews degrade to the waveform-only overlay.
    private let makePreviewProvider: () -> TranscriptionProvider
    private var previewProvider: TranscriptionProvider?
    private var previewPrepareFailed = false

    init(
        session: URLSession? = nil,
        makePreviewProvider: @escaping () -> TranscriptionProvider = {
            WhisperProvider(modelOverride: .whisperSwissGermanQ4)
        }
    ) {
        self.makePreviewProvider = makePreviewProvider
        if let session {
            self.session = session
        } else {
            let configuration = URLSessionConfiguration.ephemeral
            // Whisper on local hardware can take a while for long utterances.
            configuration.timeoutIntervalForRequest = 120
            configuration.timeoutIntervalForResource = 300
            self.session = URLSession(configuration: configuration)
        }
    }

    // MARK: - Lifecycle

    func prepare(progressHandler: ((ModelPreparationProgress) -> Void)?) async throws {
        self.isReady = false

        // Nothing to download; validate the configuration and confirm the server answers.
        let settings = SettingsStore.shared
        _ = try Self.endpointURL(baseURL: settings.customASRBaseURL)
        guard !settings.customASRModelName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw OpenAICompatibleTranscriptionError.missingModelName
        }
        try await self.verifyEndpointReachable(baseURL: settings.customASRBaseURL, apiKey: settings.customASRAPIKey)

        self.isReady = true
        DebugLogger.shared.info("OpenAICompatibleTranscriptionProvider ready", source: "OpenAICompatibleTranscriptionProvider")
    }

    /// The dictation path swallows transcription errors and returns empty text, so a wrong key
    /// or an unreachable host would surface as a silently empty dictation. Probing here moves
    /// that failure to model activation, where the error is shown to the user.
    private func verifyEndpointReachable(baseURL: String, apiKey: String) async throws {
        var request = URLRequest(url: try Self.modelsURL(baseURL: baseURL))
        request.httpMethod = "GET"
        request.timeoutInterval = 15
        let key = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        if !key.isEmpty {
            request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        }

        let (data, response) = try await self.session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else { return }
        // Servers that do not implement /models still transcribe fine, so only reject auth failures.
        guard httpResponse.statusCode == 401 || httpResponse.statusCode == 403 else { return }
        throw OpenAICompatibleTranscriptionError.serverError(
            statusCode: httpResponse.statusCode,
            message: Self.extractServerMessage(from: data)
        )
    }

    func modelsExistOnDisk() -> Bool {
        return true // Remote inference; nothing stored locally
    }

    func clearCache() async throws {
        // No local cache
    }

    // MARK: - Streaming Preview (local Swiss German Q4)

    func transcribeStreaming(_ samples: [Float]) async throws -> ASRTranscriptionResult {
        // Never route previews over HTTP: chunked re-transcription would hammer
        // the server. Preview locally when enabled and the Q4 model is
        // installed, otherwise return empty text so the overlay keeps its
        // waveform-only state.
        guard SettingsStore.shared.customASRLivePreviewEnabled else {
            // Release the ~1 GB GGUF when the user turns previews off mid-session.
            self.previewProvider = nil
            return ASRTranscriptionResult(text: "")
        }
        guard SettingsStore.SpeechModel.whisperSwissGermanQ4.isInstalled, !self.previewPrepareFailed else {
            return ASRTranscriptionResult(text: "")
        }

        let provider: TranscriptionProvider
        if let existing = previewProvider {
            provider = existing
        } else {
            provider = self.makePreviewProvider()
            self.previewProvider = provider
        }

        if !provider.isReady {
            do {
                try await provider.prepare(progressHandler: nil)
            } catch {
                // One failed load (e.g. corrupt file) must not fail every chunk.
                self.previewPrepareFailed = true
                DebugLogger.shared.error(
                    "Local preview model failed to load, previews disabled: \(error.localizedDescription)",
                    source: "OpenAICompatibleTranscriptionProvider"
                )
                return ASRTranscriptionResult(text: "")
            }
        }

        return try await provider.transcribeStreaming(samples)
    }

    // MARK: - Transcription

    func transcribe(_ samples: [Float]) async throws -> ASRTranscriptionResult {
        guard !samples.isEmpty else {
            throw OpenAICompatibleTranscriptionError.emptyAudio
        }

        let settings = SettingsStore.shared
        let request = try Self.makeRequest(
            baseURL: settings.customASRBaseURL,
            apiKey: settings.customASRAPIKey,
            modelName: settings.customASRModelName,
            language: settings.customASRLanguage,
            wavData: Self.encodeWAV(samples: samples, sampleRate: 16000)
        )

        let (data, response): (Data, URLResponse)
        do {
            (data, response) = try await self.session.data(for: request)
        } catch {
            DebugLogger.shared.error(
                "Custom ASR request failed: \(error.localizedDescription)",
                source: "OpenAICompatibleTranscriptionProvider"
            )
            throw error
        }

        guard let httpResponse = response as? HTTPURLResponse else {
            throw OpenAICompatibleTranscriptionError.invalidResponse
        }
        guard (200 ..< 300).contains(httpResponse.statusCode) else {
            throw OpenAICompatibleTranscriptionError.serverError(
                statusCode: httpResponse.statusCode,
                message: Self.extractServerMessage(from: data)
            )
        }

        guard
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let text = json["text"] as? String
        else {
            throw OpenAICompatibleTranscriptionError.invalidResponse
        }

        return ASRTranscriptionResult(text: text.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    // MARK: - Request Construction (internal for tests)

    /// The API root, with a trailing transcription path stripped so sibling
    /// endpoints such as `/models` can be derived from the same setting.
    static func normalizedBaseURL(_ baseURL: String) throws -> URL {
        let trimmed = baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: trimmed), url.scheme != nil, url.host != nil else {
            throw OpenAICompatibleTranscriptionError.invalidBaseURL(baseURL)
        }
        guard url.path.hasSuffix("/audio/transcriptions") else { return url }
        return url.deletingLastPathComponent().deletingLastPathComponent()
    }

    /// Resolves the transcription endpoint from the configured base URL.
    static func endpointURL(baseURL: String) throws -> URL {
        try self.normalizedBaseURL(baseURL).appendingPathComponent("audio/transcriptions")
    }

    /// Resolves the model-listing endpoint, used as a reachability and auth probe.
    static func modelsURL(baseURL: String) throws -> URL {
        try self.normalizedBaseURL(baseURL).appendingPathComponent("models")
    }

    static func makeRequest(
        baseURL: String,
        apiKey: String,
        modelName: String,
        language: String,
        wavData: Data
    ) throws -> URLRequest {
        let model = modelName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !model.isEmpty else {
            throw OpenAICompatibleTranscriptionError.missingModelName
        }

        var request = URLRequest(url: try self.endpointURL(baseURL: baseURL))
        request.httpMethod = "POST"

        let key = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        if !key.isEmpty {
            request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        }

        let boundary = "fluidvoice-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()
        func appendField(name: String, value: String) {
            body.append(Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"\(name)\"\r\n\r\n\(value)\r\n".utf8))
        }
        appendField(name: "model", value: model)
        appendField(name: "response_format", value: "json")
        let languageCode = language.trimmingCharacters(in: .whitespacesAndNewlines)
        if !languageCode.isEmpty {
            appendField(name: "language", value: languageCode)
        }
        body.append(Data((
            "--\(boundary)\r\n" +
            "Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n" +
            "Content-Type: audio/wav\r\n\r\n"
        ).utf8))
        body.append(wavData)
        body.append(Data("\r\n--\(boundary)--\r\n".utf8))
        request.httpBody = body

        return request
    }

    /// Encodes 16 kHz mono float samples as a 16-bit PCM WAV file.
    static func encodeWAV(samples: [Float], sampleRate: Int) -> Data {
        let bytesPerSample = 2
        let dataSize = samples.count * bytesPerSample

        var data = Data(capacity: 44 + dataSize)

        func appendUInt32(_ value: UInt32) {
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }
        func appendUInt16(_ value: UInt16) {
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }

        data.append(Data("RIFF".utf8))
        appendUInt32(UInt32(36 + dataSize))
        data.append(Data("WAVE".utf8))

        data.append(Data("fmt ".utf8))
        appendUInt32(16) // PCM fmt chunk size
        appendUInt16(1) // PCM format
        appendUInt16(1) // mono
        appendUInt32(UInt32(sampleRate))
        appendUInt32(UInt32(sampleRate * bytesPerSample)) // byte rate
        appendUInt16(UInt16(bytesPerSample)) // block align
        appendUInt16(16) // bits per sample

        data.append(Data("data".utf8))
        appendUInt32(UInt32(dataSize))

        // Convert in one pass and append the payload in a single write; a per-sample
        // append costs millions of calls on a multi-minute dictation.
        let pcm = [Int16](unsafeUninitializedCapacity: samples.count) { buffer, initializedCount in
            for (index, sample) in samples.enumerated() {
                let clamped = max(-1.0, min(1.0, sample))
                buffer[index] = Int16(clamped * Float(Int16.max)).littleEndian
            }
            initializedCount = samples.count
        }
        pcm.withUnsafeBytes { data.append(contentsOf: $0) }

        return data
    }

    private static func extractServerMessage(from data: Data) -> String? {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return String(data: data.prefix(200), encoding: .utf8)
        }
        if let error = json["error"] as? [String: Any], let message = error["message"] as? String {
            return message
        }
        return json["message"] as? String
    }
}
