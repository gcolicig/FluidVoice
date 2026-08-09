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
        case .serverError(let statusCode, let message):
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

    init(session: URLSession? = nil) {
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
        // Nothing to download; just validate the configuration is usable.
        _ = try Self.endpointURL(baseURL: SettingsStore.shared.customASRBaseURL)
        guard !SettingsStore.shared.customASRModelName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw OpenAICompatibleTranscriptionError.missingModelName
        }
        self.isReady = true
        DebugLogger.shared.info("OpenAICompatibleTranscriptionProvider ready", source: "OpenAICompatibleTranscriptionProvider")
    }

    func modelsExistOnDisk() -> Bool {
        return true // Remote inference; nothing stored locally
    }

    func clearCache() async throws {
        // No local cache
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

    /// Resolves the transcription endpoint from the configured base URL.
    /// Appends `/audio/transcriptions` unless the URL already points at it.
    static func endpointURL(baseURL: String) throws -> URL {
        let trimmed = baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
        guard var url = URL(string: trimmed), url.scheme != nil, url.host != nil else {
            throw OpenAICompatibleTranscriptionError.invalidBaseURL(baseURL)
        }
        if !url.path.hasSuffix("/audio/transcriptions") {
            url.appendPathComponent("audio/transcriptions")
        }
        return url
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
        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            let value = Int16(clamped * Float(Int16.max))
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }

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
