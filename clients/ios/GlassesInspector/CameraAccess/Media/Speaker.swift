/*
 * Speaker.swift — Glasses Inspector addition.
 *
 * Speaks text from the Mac (Claude's analysis) through the glasses. Two paths:
 *   1. PCM audio streamed from the Mac (ElevenLabs voice), played through an AVAudioEngine
 *      player node as chunks arrive, so speech starts before the sentence has fully rendered.
 *   2. Apple's on-device synthesizer, used when the Mac has no TTS key or a request fails.
 * The glasses are the phone's Bluetooth audio output, so a playback-category session routes
 * both to them over A2DP. While the sample is recording it holds a playAndRecord session; we
 * leave that alone and play through it as-is.
 */

import AVFoundation
import Foundation
import Observation

@Observable
@MainActor
final class Speaker: NSObject, AVSpeechSynthesizerDelegate {
  private let synth = AVSpeechSynthesizer()
  private let pcm = PCMStreamPlayer()
  private(set) var synthSpeaking = false
  var isSpeaking: Bool { synthSpeaking || pcm.isPlaying }
  var rate: Float = AVSpeechUtteranceDefaultSpeechRate
  var language: String = "en-US"

  override init() {
    super.init()
    synth.delegate = self
  }

  // MARK: Apple voice (fallback)

  func speak(_ text: String) {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return }
    configureSession()
    let u = AVSpeechUtterance(string: trimmed)
    u.rate = rate
    u.voice = Self.preferredVoice(language: language)
    u.postUtteranceDelay = 0.15
    synth.speak(u)  // queues behind anything already speaking
    synthSpeaking = true
  }

  /// Best installed voice for the language: premium, then enhanced, then whatever exists.
  /// Premium voices are downloaded in Settings > Accessibility > Spoken Content > Voices.
  static func preferredVoice(language: String) -> AVSpeechSynthesisVoice? {
    let candidates = AVSpeechSynthesisVoice.speechVoices().filter { $0.language == language }
    return candidates.first { $0.quality == .premium }
      ?? candidates.first { $0.quality == .enhanced }
      ?? AVSpeechSynthesisVoice(language: language)
  }

  // MARK: Streamed PCM from the Mac

  /// Queue a chunk of 16-bit mono PCM for the utterance `id`. Chunks arrive in order.
  func playPCM(id: Int, data: Data, sampleRate: Double) {
    configureSession()
    pcm.enqueue(id: id, data: data, sampleRate: sampleRate)
  }

  /// Marks the end of an utterance's audio so `isSpeaking` can drop when playback drains.
  func finishPCM(id: Int) {
    pcm.finish(id: id)
  }

  func stop() {
    synth.stopSpeaking(at: .immediate)
    synthSpeaking = false
    pcm.stop()
  }

  private func configureSession() {
    let session = AVAudioSession.sharedInstance()
    // Recording owns the session (playAndRecord + HFP); don't fight it.
    guard session.category != .playAndRecord else { return }
    if session.category != .playback {
      try? session.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
    }
    try? session.setActive(true)
  }

  nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
    Task { @MainActor in
      if !synthesizer.isSpeaking { self.synthSpeaking = false }
    }
  }

  nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
    Task { @MainActor in self.synthSpeaking = false }
  }
}

/// Plays 16-bit mono PCM chunks as they arrive. One player node, buffers scheduled in order;
/// the engine's mixer resamples to whatever the Bluetooth route wants.
@Observable
@MainActor
final class PCMStreamPlayer {
  private let engine = AVAudioEngine()
  private let node = AVAudioPlayerNode()
  private var format: AVAudioFormat?
  private var outstanding = 0          // buffers scheduled but not yet played
  private var openIDs = Set<Int>()     // utterances still receiving chunks
  private(set) var isPlaying = false

  init() {
    engine.attach(node)
    NotificationCenter.default.addObserver(
      forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main
    ) { [weak self] _ in
      Task { @MainActor in self?.restart() }
    }
  }

  func enqueue(id: Int, data: Data, sampleRate: Double) {
    guard data.count >= 2 else { return }
    if format == nil || format!.sampleRate != sampleRate {
      format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)
      engine.disconnectNodeOutput(node)
      engine.connect(node, to: engine.mainMixerNode, format: format)
    }
    guard let format, let buf = Self.floatBuffer(from: data, format: format) else { return }
    if !engine.isRunning {
      engine.prepare()
      do { try engine.start() } catch { return }
    }
    openIDs.insert(id)
    outstanding += 1
    isPlaying = true
    node.scheduleBuffer(buf, completionCallbackType: .dataPlayedBack) { [weak self] _ in
      Task { @MainActor in self?.bufferDone() }
    }
    if !node.isPlaying { node.play() }
  }

  func finish(id: Int) {
    openIDs.remove(id)
    updatePlaying()
  }

  func stop() {
    node.stop()
    node.reset()
    outstanding = 0
    openIDs.removeAll()
    isPlaying = false
  }

  private func bufferDone() {
    outstanding = max(0, outstanding - 1)
    updatePlaying()
  }

  private func updatePlaying() {
    isPlaying = outstanding > 0 || !openIDs.isEmpty
  }

  private func restart() {
    // Route changed (glasses connected or dropped). Restart the engine; scheduled audio is lost,
    // which is acceptable for short narration sentences.
    guard outstanding > 0 || !openIDs.isEmpty else { return }
    engine.prepare()
    try? engine.start()
    if !node.isPlaying { node.play() }
  }

  /// Int16 little-endian mono -> Float32 buffer the mixer accepts.
  private static func floatBuffer(from data: Data, format: AVAudioFormat) -> AVAudioPCMBuffer? {
    let frames = AVAudioFrameCount(data.count / 2)
    guard frames > 0, let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames),
      let out = buf.floatChannelData?[0] else { return nil }
    buf.frameLength = frames
    data.withUnsafeBytes { raw in
      let src = raw.bindMemory(to: Int16.self)
      for i in 0..<Int(frames) {
        out[i] = Float(Int16(littleEndian: src[i])) / 32768.0
      }
    }
    return buf
  }
}
