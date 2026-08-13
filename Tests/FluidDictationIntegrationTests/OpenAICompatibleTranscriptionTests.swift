@testable import FluidVoice_Debug
import XCTest

final class OpenAICompatibleTranscriptionTests: XCTestCase {
    // MARK: - Endpoint URL

    func testEndpointAppendsAudioTranscriptionsToV1BaseURL() throws {
        let url = try OpenAICompatibleTranscriptionProvider.endpointURL(baseURL: "http://127.0.0.1:8888/v1")
        XCTAssertEqual(url.absoluteString, "http://127.0.0.1:8888/v1/audio/transcriptions")
    }

    func testEndpointKeepsFullPathUntouched() throws {
        let url = try OpenAICompatibleTranscriptionProvider.endpointURL(
            baseURL: "http://localhost:8888/v1/audio/transcriptions"
        )
        XCTAssertEqual(url.absoluteString, "http://localhost:8888/v1/audio/transcriptions")
    }

    func testEndpointRejectsInvalidURL() {
        XCTAssertThrowsError(try OpenAICompatibleTranscriptionProvider.endpointURL(baseURL: "not a url"))
    }

    func testModelsURLIsDerivedFromApiRoot() throws {
        let fromRoot = try OpenAICompatibleTranscriptionProvider.modelsURL(baseURL: "http://127.0.0.1:8888/v1")
        XCTAssertEqual(fromRoot.absoluteString, "http://127.0.0.1:8888/v1/models")

        // A base URL that already carries the transcription path must resolve to the
        // same sibling endpoint rather than nesting under it.
        let fromFullPath = try OpenAICompatibleTranscriptionProvider.modelsURL(
            baseURL: "http://127.0.0.1:8888/v1/audio/transcriptions"
        )
        XCTAssertEqual(fromFullPath.absoluteString, "http://127.0.0.1:8888/v1/models")
    }

    // MARK: - Request Construction

    func testRequestContainsAuthHeaderModelFieldAndWAV() throws {
        let wav = OpenAICompatibleTranscriptionProvider.encodeWAV(samples: [0.0, 0.5, -0.5], sampleRate: 16000)
        let request = try OpenAICompatibleTranscriptionProvider.makeRequest(
            baseURL: "http://127.0.0.1:8888/v1",
            apiKey: "test-key",
            modelName: "gcoli/whisper-large-v3-swiss-german-mlx-fp16",
            language: "de",
            wavData: wav
        )

        XCTAssertEqual(request.httpMethod, "POST")
        XCTAssertEqual(request.value(forHTTPHeaderField: "Authorization"), "Bearer test-key")
        let contentType = try XCTUnwrap(request.value(forHTTPHeaderField: "Content-Type"))
        XCTAssertTrue(contentType.hasPrefix("multipart/form-data; boundary="))

        // The body mixes text fields with raw WAV bytes, so it is not decodable as UTF-8.
        // Match the field markers as byte sequences instead.
        let body = try XCTUnwrap(request.httpBody)
        XCTAssertTrue(body.contains(text: "name=\"model\"\r\n\r\ngcoli/whisper-large-v3-swiss-german-mlx-fp16"))
        XCTAssertTrue(body.contains(text: "name=\"language\"\r\n\r\nde"))
        XCTAssertTrue(body.contains(text: "filename=\"audio.wav\""))
        XCTAssertTrue(body.contains(text: "RIFF"))
    }

    func testRequestOmitsAuthAndLanguageWhenEmpty() throws {
        let request = try OpenAICompatibleTranscriptionProvider.makeRequest(
            baseURL: "http://127.0.0.1:8888/v1",
            apiKey: "  ",
            modelName: "whisper-large-v3",
            language: "",
            wavData: Data()
        )
        XCTAssertNil(request.value(forHTTPHeaderField: "Authorization"))
        let body = try XCTUnwrap(request.httpBody)
        XCTAssertFalse(body.contains(text: "name=\"language\""))
        XCTAssertTrue(body.contains(text: "name=\"model\"\r\n\r\nwhisper-large-v3"))
    }

    func testRequestRejectsEmptyModelName() {
        XCTAssertThrowsError(
            try OpenAICompatibleTranscriptionProvider.makeRequest(
                baseURL: "http://127.0.0.1:8888/v1",
                apiKey: "k",
                modelName: "   ",
                language: "",
                wavData: Data()
            )
        )
    }

    // MARK: - WAV Encoding

    func testWAVHeaderFieldsAndSampleConversion() {
        let samples: [Float] = [0.0, 1.0, -1.0, 2.0] // 2.0 must clamp to 1.0
        let wav = OpenAICompatibleTranscriptionProvider.encodeWAV(samples: samples, sampleRate: 16000)

        XCTAssertEqual(wav.count, 44 + samples.count * 2)
        XCTAssertEqual(String(bytes: wav[0 ..< 4], encoding: .utf8), "RIFF")
        XCTAssertEqual(String(bytes: wav[8 ..< 12], encoding: .utf8), "WAVE")
        XCTAssertEqual(String(bytes: wav[36 ..< 40], encoding: .utf8), "data")

        func int16(at offset: Int) -> Int16 {
            Int16(littleEndian: wav[offset ..< offset + 2].withUnsafeBytes { $0.loadUnaligned(as: Int16.self) })
        }
        XCTAssertEqual(int16(at: 44), 0)
        XCTAssertEqual(int16(at: 46), Int16.max)
        XCTAssertEqual(int16(at: 48), -Int16.max) // clamped to -1.0
        XCTAssertEqual(int16(at: 50), Int16.max) // clamped from 2.0

        // Sample rate at offset 24 (little-endian UInt32)
        let sampleRate = wav[24 ..< 28].withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(as: UInt32.self)) }
        XCTAssertEqual(sampleRate, 16000)
    }

    func testWAVEncodingEmptySamples() {
        let wav = OpenAICompatibleTranscriptionProvider.encodeWAV(samples: [], sampleRate: 16000)
        XCTAssertEqual(wav.count, 44)
    }

    // MARK: - Swiss German Q4 Preview Model

    func testSwissGermanQ4ModelMetadata() {
        let model = SettingsStore.SpeechModel.whisperSwissGermanQ4
        // The GGUF installed-check compares the on-disk size against this exact
        // byte count; it must match the file published on Hugging Face.
        XCTAssertEqual(model.expectedDownloadBytes, 901_544_064)
        XCTAssertEqual(model.whisperModelFile, "whisper-large-v3-swiss-german-Q4_0.gguf")
        XCTAssertEqual(
            model.whisperModelDownloadOverrideURL?.absoluteString,
            "https://huggingface.co/gcoli/whisper-large-v3-swiss-german-gguf-q4_0/resolve/main/whisper-large-v3-swiss-german-Q4_0.gguf"
        )
        XCTAssertTrue(model.isWhisperModel)
        XCTAssertFalse(model.supportsStreaming)
        // Forced decode language prevents autodetect hallucinations on short
        // preview chunks; other whisper models keep autodetection.
        XCTAssertEqual(model.whisperForcedLanguage, "de")
        XCTAssertNil(SettingsStore.SpeechModel.whisperLarge.whisperForcedLanguage)
    }

    func testCustomServerModelStreamsViaLocalPreview() {
        // The custom server model advertises streaming (served by the local Q4
        // preview engine) while the toggle is on; the remote round-trip itself
        // stays batch-only either way.
        let settings = SettingsStore.shared
        let original = settings.customASRLivePreviewEnabled
        defer { settings.customASRLivePreviewEnabled = original }

        settings.customASRLivePreviewEnabled = true
        XCTAssertTrue(SettingsStore.SpeechModel.customOpenAICompatible.supportsStreaming)
        XCTAssertEqual(SettingsStore.SpeechModel.customOpenAICompatible.streamingPreviewIntervalSeconds, 1.0)

        settings.customASRLivePreviewEnabled = false
        XCTAssertFalse(SettingsStore.SpeechModel.customOpenAICompatible.supportsStreaming)
    }

    func testStreamingReturnsEmptyResultWhenPreviewDisabled() async throws {
        final class UnexpectedProvider: TranscriptionProvider {
            var name: String { "unexpected" }
            var isAvailable: Bool { true }
            var isReady: Bool { false }
            func prepare(progressHandler: ((ModelPreparationProgress) -> Void)?) async throws {
                XCTFail("preview provider must not be prepared while previews are disabled")
            }

            func transcribe(_ samples: [Float]) async throws -> ASRTranscriptionResult {
                XCTFail("preview provider must not transcribe while previews are disabled")
                return ASRTranscriptionResult(text: "unexpected")
            }
        }

        let settings = SettingsStore.shared
        let original = settings.customASRLivePreviewEnabled
        defer { settings.customASRLivePreviewEnabled = original }
        settings.customASRLivePreviewEnabled = false

        let provider = OpenAICompatibleTranscriptionProvider(makePreviewProvider: { UnexpectedProvider() })
        let result = try await provider.transcribeStreaming([0.0, 0.1, -0.1])
        XCTAssertEqual(result.text, "")
    }

    func testStreamingReturnsEmptyResultWithoutInstalledPreviewModel() async throws {
        // Preview provider factory that must never be invoked when the Q4 model
        // is absent — streaming degrades to empty text without any network call.
        final class FailingProvider: TranscriptionProvider {
            var name: String { "failing" }
            var isAvailable: Bool { true }
            var isReady: Bool { false }
            func prepare(progressHandler: ((ModelPreparationProgress) -> Void)?) async throws {
                XCTFail("preview provider must not be prepared when Q4 model is not installed")
            }

            func transcribe(_ samples: [Float]) async throws -> ASRTranscriptionResult {
                XCTFail("preview provider must not transcribe when Q4 model is not installed")
                return ASRTranscriptionResult(text: "unexpected")
            }
        }

        try XCTSkipIf(
            SettingsStore.SpeechModel.whisperSwissGermanQ4.isInstalled,
            "Q4 model is installed on this machine; the degradation path is not reachable"
        )

        let settings = SettingsStore.shared
        let original = settings.customASRLivePreviewEnabled
        defer { settings.customASRLivePreviewEnabled = original }
        settings.customASRLivePreviewEnabled = true

        let provider = OpenAICompatibleTranscriptionProvider(makePreviewProvider: { FailingProvider() })
        let result = try await provider.transcribeStreaming([0.0, 0.1, -0.1])
        XCTAssertEqual(result.text, "")
    }
}

private extension Data {
    /// Byte-level match, for multipart bodies that mix text fields with binary payloads.
    func contains(text: String) -> Bool {
        self.range(of: Data(text.utf8)) != nil
    }
}
