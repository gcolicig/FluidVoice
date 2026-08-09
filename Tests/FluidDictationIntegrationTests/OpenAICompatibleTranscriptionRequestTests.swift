import XCTest

@testable import FluidVoice_Debug

final class OpenAICompatibleTranscriptionRequestTests: XCTestCase {
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

        let body = try XCTUnwrap(request.httpBody)
        let bodyString = String(decoding: body, as: UTF8.self)
        XCTAssertTrue(bodyString.contains("name=\"model\"\r\n\r\ngcoli/whisper-large-v3-swiss-german-mlx-fp16"))
        XCTAssertTrue(bodyString.contains("name=\"language\"\r\n\r\nde"))
        XCTAssertTrue(bodyString.contains("filename=\"audio.wav\""))
        XCTAssertNotNil(body.range(of: Data("RIFF".utf8)))
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
        let bodyString = String(decoding: try XCTUnwrap(request.httpBody), as: UTF8.self)
        XCTAssertFalse(bodyString.contains("name=\"language\""))
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
        XCTAssertEqual(String(decoding: wav[0 ..< 4], as: UTF8.self), "RIFF")
        XCTAssertEqual(String(decoding: wav[8 ..< 12], as: UTF8.self), "WAVE")
        XCTAssertEqual(String(decoding: wav[36 ..< 40], as: UTF8.self), "data")

        func int16(at offset: Int) -> Int16 {
            Int16(littleEndian: wav[offset ..< offset + 2].withUnsafeBytes { $0.load(as: Int16.self) })
        }
        XCTAssertEqual(int16(at: 44), 0)
        XCTAssertEqual(int16(at: 46), Int16.max)
        XCTAssertEqual(int16(at: 48), -Int16.max) // clamped to -1.0
        XCTAssertEqual(int16(at: 50), Int16.max) // clamped from 2.0

        // Sample rate at offset 24 (little-endian UInt32)
        let sampleRate = wav[24 ..< 28].withUnsafeBytes { UInt32(littleEndian: $0.load(as: UInt32.self)) }
        XCTAssertEqual(sampleRate, 16000)
    }

    func testWAVEncodingEmptySamples() {
        let wav = OpenAICompatibleTranscriptionProvider.encodeWAV(samples: [], sampleRate: 16000)
        XCTAssertEqual(wav.count, 44)
    }
}
