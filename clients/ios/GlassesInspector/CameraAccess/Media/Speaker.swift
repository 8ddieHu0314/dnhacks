/*
 * Speaker.swift — Glasses Inspector addition.
 *
 * Speaks text from the Mac (Claude's analysis) with Apple's on-device synthesizer. The
 * glasses are the phone's Bluetooth audio output, so a playback-category session routes
 * speech to them over A2DP. While the sample is recording it holds a playAndRecord
 * session; we leave that alone and speak through it as-is.
 */

import AVFoundation
import Foundation
import Observation

@Observable
@MainActor
final class Speaker: NSObject, AVSpeechSynthesizerDelegate {
  private let synth = AVSpeechSynthesizer()
  private(set) var isSpeaking = false
  var rate: Float = AVSpeechUtteranceDefaultSpeechRate
  var language: String = "en-US"

  override init() {
    super.init()
    synth.delegate = self
  }

  func speak(_ text: String) {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return }
    configureSession()
    let u = AVSpeechUtterance(string: trimmed)
    u.rate = rate
    u.voice = AVSpeechSynthesisVoice(language: language)
    u.postUtteranceDelay = 0.15
    synth.speak(u)  // queues behind anything already speaking
    isSpeaking = true
  }

  func stop() {
    synth.stopSpeaking(at: .immediate)
    isSpeaking = false
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
      if !synthesizer.isSpeaking { self.isSpeaking = false }
    }
  }

  nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
    Task { @MainActor in self.isSpeaking = false }
  }
}
