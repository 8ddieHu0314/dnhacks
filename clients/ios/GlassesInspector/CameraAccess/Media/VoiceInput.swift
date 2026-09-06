/*
 * VoiceInput.swift — Glasses Inspector addition.
 *
 * Listens to the wearer through the glasses microphone (Bluetooth HFP) or the phone's own
 * microphone, transcribes on the phone with Apple's speech recognizer, and hands finished
 * utterances to FrameRelay, which sends them to the Mac as {"type":"ask","text":...}.
 *
 * Why on the phone: the DAT SDK exposes no audio API. The glasses are a Bluetooth headset, so
 * the phone's audio session is the only way to reach their mic, and recognizing here keeps
 * audio off the phone->Mac link, which is already the bandwidth ceiling for frames.
 *
 * Endpointing: the recognizer streams partial results; an utterance ends after `silence`
 * seconds without a change, then a fresh request starts so the next words are not lost.
 * Wake-word mode forwards only what follows the wake word ("inspector, how many pins?");
 * the wake word alone arms the next utterance for a few seconds.
 *
 * Audio session: listening needs .playAndRecord. With the glasses mic that means Bluetooth HFP
 * in both directions (16 kHz mono, so the ElevenLabs voice sounds narrower while listening);
 * with the phone mic the glasses keep A2DP output. Speaker.swift leaves a .playAndRecord
 * session alone, so playback keeps working either way. Turning listening off restores .playback.
 *
 * Echo: while the glasses are playing Claude's answer the mic hears it too. Utterances that end
 * while output is playing are dropped unless they are a stop word, and the recognizer context
 * is reset when playback ends so the answer's words never leak into the next question.
 */

import AVFoundation
import Foundation
import Observation
import Speech

/// Holds the live recognition request for the audio tap, which runs off the main actor.
private final class RequestBox: @unchecked Sendable {
  private let lock = NSLock()
  private var request: SFSpeechAudioBufferRecognitionRequest?
  func set(_ r: SFSpeechAudioBufferRecognitionRequest?) { lock.lock(); request = r; lock.unlock() }
  func append(_ buffer: AVAudioPCMBuffer) {
    guard buffer.frameLength > 0 else { return }   // empty buffers arrive while a Bluetooth route switches
    lock.lock(); let r = request; lock.unlock()
    r?.append(buffer)
  }
}

@Observable
@MainActor
final class VoiceInput {
  enum Mode: String, CaseIterable { case off, wake, always }
  enum Mic: String, CaseIterable { case glasses, phone }

  /// off: mic closed. wake: only sentences addressed with the wake word. always: every sentence.
  var mode: Mode {
    didSet {
      UserDefaults.standard.set(mode.rawValue, forKey: "voiceInputMode")
      if mode == .off { stop() } else if !isListening { Task { await start() } }
    }
  }
  /// glasses: Bluetooth HFP mic on the glasses. phone: the iPhone's built-in mic, A2DP output stays.
  var mic: Mic {
    didSet {
      UserDefaults.standard.set(mic.rawValue, forKey: "voiceMic")
      if isListening { restartEngine(reconfigure: true) }
    }
  }
  var wakeWord: String {
    didSet { UserDefaults.standard.set(wakeWord, forKey: "wakeWord") }
  }
  /// Seconds of no new words before an utterance is considered finished.
  var silence: TimeInterval = 0.9
  /// Words that stop playback even while the glasses are talking.
  static let stopWords = ["stop", "hush", "quiet", "shut up", "enough", "silence", "cancel"]

  private(set) var isListening = false
  /// off | starting | listening | armed | denied | unavailable | error: ...
  private(set) var status = "off"
  /// Live transcript of the sentence being spoken, for the on-screen caption.
  private(set) var partial = ""
  private(set) var lastUtterance = ""
  /// When the sentence being transcribed began (first partial result). The Mac uses it to pick
  /// the camera frames from the moment the wearer was speaking.
  private(set) var lastUtteranceStart: Date?
  /// Input port in use, for the settings sheet ("Ray-Ban Meta (BluetoothHFP)").
  private(set) var route = ""
  private(set) var utterances = 0

  /// Final text to act on (wake word already stripped).
  var onUtterance: ((String) -> Void)?
  /// The wake word was heard; a good moment to stop playback so the question is not talked over.
  var onWake: (() -> Void)?
  /// Whether the glasses are currently playing speech (echo guard). Set by FrameRelay.
  var isOutputPlaying: () -> Bool = { false }

  /// Rebuilt on every (re)start: an engine that already touched its input node keeps that
  /// route's format, and installing a tap against a stale format is an uncatchable crash.
  private var engine = AVAudioEngine()
  private var engineObserver: NSObjectProtocol?
  private var restarting = false
  private let box = RequestBox()
  private var recognizer: SFSpeechRecognizer?
  private var request: SFSpeechAudioBufferRecognitionRequest?
  private var task: SFSpeechRecognitionTask?
  private var generation = 0          // identifies the live task; results from older tasks are ignored
  private var silenceTimer: Task<Void, Never>?
  private var echoTask: Task<Void, Never>?
  private var armedUntil: Date = .distantPast
  private var lastText = ""
  private var observers: [NSObjectProtocol] = []
  /// Bumped by stop(); a start() that resumes from an await after it must abandon its work.
  private var startGen = 0
  /// AVAudioSession and AVAudioEngine calls block, and called from Swift concurrency they log
  /// "unsafeForcedSync called from Swift Concurrent context" and stall the main thread while a
  /// Bluetooth route switches. They run on this serial queue instead.
  private let audioQueue = DispatchQueue(label: "GlassesInspector.voice.audio", qos: .userInitiated)

  private func onAudioQueue<T>(_ work: @escaping () throws -> T) async throws -> T {
    try await withCheckedThrowingContinuation { (c: CheckedContinuation<T, Error>) in
      audioQueue.async {
        do { c.resume(returning: try work()) } catch { c.resume(throwing: error) }
      }
    }
  }

  init() {
    let d = UserDefaults.standard
    mode = Mode(rawValue: d.string(forKey: "voiceInputMode") ?? "") ?? .off
    // Phone mic by default: the glasses mic opens a Bluetooth HFP link that shares the radio with
    // the DAT video stream, and while the glasses spoke the video decoder saw corrupt frames
    // (freeze, then watchdog restarts). With the phone mic the glasses keep A2DP output only.
    mic = Mic(rawValue: d.string(forKey: "voiceMic") ?? "") ?? .phone
    wakeWord = d.string(forKey: "wakeWord") ?? "inspector"
  }

  /// Start listening if the saved mode says so. Called once the relay is up, so the permission
  /// prompts appear on the camera screen rather than at launch.
  func startIfEnabled() {
    if mode != .off, !isListening { Task { await start() } }
  }

  // MARK: Lifecycle

  func start() async {
    guard !isListening else { return }
    status = "starting"
    guard await requestPermissions() else { status = "denied"; return }
    guard let r = SFSpeechRecognizer(locale: Locale(identifier: "en-US")), r.isAvailable else {
      status = "unavailable"
      return
    }
    recognizer = r
    let gen = startGen
    do {
      let mic = self.mic
      route = try await onAudioQueue { try Self.applySession(mic: mic) }
      await waitForRoute()
      guard gen == startGen, mode != .off else { return }   // stopped while we were starting
      try await startEngine()
    } catch {
      status = "error: \(error.localizedDescription)"
      return
    }
    guard gen == startGen, mode != .off else { tearDownEngine(); return }
    isListening = true
    status = "listening"
    startRequest()
    observeRoute()
    startEchoGuard()
  }

  func stop() {
    silenceTimer?.cancel(); silenceTimer = nil
    echoTask?.cancel(); echoTask = nil
    for o in observers { NotificationCenter.default.removeObserver(o) }
    observers.removeAll()
    generation += 1
    task?.cancel(); task = nil
    box.set(nil)                 // stop feeding the request before ending it
    request?.endAudio(); request = nil
    startGen += 1
    tearDownEngine()
    isListening = false
    partial = ""
    lastText = ""
    status = "off"
    // Hand the session back to playback so the glasses return to A2DP, unless the sample's
    // video recording owns the session right now.
    audioQueue.async {
      let s = AVAudioSession.sharedInstance()
      guard s.mode != .videoRecording else { return }
      try? s.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
      try? s.setActive(true)
    }
  }

  private func requestPermissions() async -> Bool {
    guard await AVAudioApplication.requestRecordPermission() else { return false }
    let speech = await withCheckedContinuation { (c: CheckedContinuation<SFSpeechRecognizerAuthorizationStatus, Never>) in
      SFSpeechRecognizer.requestAuthorization { c.resume(returning: $0) }
    }
    return speech == .authorized
  }

  /// Session category for the chosen mic. Runs on the audio queue; returns the input route.
  nonisolated private static func applySession(mic: Mic) throws -> String {
    let s = AVAudioSession.sharedInstance()
    switch mic {
    case .glasses:
      try s.setCategory(.playAndRecord, mode: .default, options: [.allowBluetoothHFP])
      if let hfp = s.availableInputs?.first(where: { $0.portType == .bluetoothHFP }) {
        try? s.setPreferredInput(hfp)
      }
    case .phone:
      try s.setCategory(.playAndRecord, mode: .default, options: [.allowBluetoothA2DP, .defaultToSpeaker])
      if let builtIn = s.availableInputs?.first(where: { $0.portType == .builtInMic }) {
        try? s.setPreferredInput(builtIn)
      }
    }
    try s.setActive(true)
    return s.currentRoute.inputs.map { "\($0.portName) (\($0.portType.rawValue))" }.joined(separator: ", ")
  }

  nonisolated private static func err(_ text: String) -> Error {
    NSError(domain: "VoiceInput", code: 1, userInfo: [NSLocalizedDescriptionKey: text])
  }

  /// Bluetooth HFP comes up asynchronously after setActive(true). Give the route up to 2 s to
  /// show the mic we asked for, so the engine below is built against the final hardware format.
  private func waitForRoute() async {
    let s = AVAudioSession.sharedInstance()
    let wanted: AVAudioSession.Port = mic == .glasses ? .bluetoothHFP : .builtInMic
    for _ in 0..<20 {
      if s.isInputAvailable, s.currentRoute.inputs.contains(where: { $0.portType == wanted }) { break }
      try? await Task.sleep(for: .milliseconds(100))
    }
    // One more beat for the hardware sample rate to follow the route.
    try? await Task.sleep(for: .milliseconds(150))
    route = s.currentRoute.inputs.map { "\($0.portName) (\($0.portType.rawValue))" }.joined(separator: ", ")
  }

  private func tearDownEngine() {
    if let o = engineObserver { NotificationCenter.default.removeObserver(o); engineObserver = nil }
    let old = engine
    audioQueue.async {
      if old.isRunning { old.stop() }
      old.inputNode.removeTap(onBus: 0)
    }
  }

  /// Builds and starts a fresh engine with the mic tap. Runs on the audio queue.
  nonisolated private static func makeEngine(box: RequestBox) throws -> AVAudioEngine {
    let session = AVAudioSession.sharedInstance()
    guard session.isInputAvailable else { throw err("no microphone on the current route") }
    let engine = AVAudioEngine()
    let input = engine.inputNode
    // The hardware side of the input node. Zero means the route has no usable mic yet (or the
    // Bluetooth link is still switching); a tap installed then throws an uncatchable
    // Objective-C exception, so bail out here instead and let the route-change path retry.
    let hw = input.inputFormat(forBus: 0)
    guard hw.sampleRate > 0, hw.channelCount > 0 else { throw err("microphone route not ready") }
    // format: nil makes the tap adopt the node's current format inside the call. Passing a
    // format read a moment earlier is what crashes when the route settles in between.
    input.installTap(onBus: 0, bufferSize: 1024, format: nil) { buffer, _ in
      box.append(buffer)
    }
    engine.prepare()
    try engine.start()
    return engine
  }

  private func startEngine() async throws {
    tearDownEngine()
    let box = self.box
    let fresh = try await onAudioQueue { try Self.makeEngine(box: box) }
    engine = fresh
    engineObserver = NotificationCenter.default.addObserver(
      forName: .AVAudioEngineConfigurationChange, object: fresh, queue: .main
    ) { [weak self] _ in
      Task { @MainActor [weak self] in self?.restartEngine(reconfigure: false) }
    }
  }

  /// Route or engine configuration changed (glasses connected or dropped), or the mic source
  /// was switched: rebuild the engine against the new route and start a fresh request.
  /// `reconfigure` re-applies the session category (only needed for a mic-source change;
  /// doing it on every route notification would trigger further route changes).
  private func restartEngine(reconfigure: Bool) {
    guard isListening, !restarting else { return }
    restarting = true
    Task { [weak self] in
      defer { self?.restarting = false }
      guard let self else { return }
      self.tearDownEngine()
      do {
        if reconfigure {
          let mic = self.mic
          self.route = try await self.onAudioQueue { try Self.applySession(mic: mic) }
        }
        await self.waitForRoute()
        guard self.isListening else { return }
        try await self.startEngine()
        self.startRequest()
        self.status = "listening"
      } catch {
        self.status = "error: \(error.localizedDescription)"
      }
    }
  }

  private func observeRoute() {
    let center = NotificationCenter.default
    observers.append(center.addObserver(forName: AVAudioSession.routeChangeNotification, object: nil, queue: .main) { [weak self] n in
      let reason = (n.userInfo?[AVAudioSessionRouteChangeReasonKey] as? UInt).flatMap(AVAudioSession.RouteChangeReason.init)
      Task { @MainActor [weak self] in
        switch reason {
        case .newDeviceAvailable, .oldDeviceUnavailable, .override, .routeConfigurationChange:
          self?.restartEngine(reconfigure: false)
        default:
          break
        }
      }
    })
    observers.append(center.addObserver(forName: AVAudioSession.interruptionNotification, object: nil, queue: .main) { [weak self] n in
      let type = (n.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt).flatMap(AVAudioSession.InterruptionType.init)
      Task { @MainActor [weak self] in
        if type == .ended { self?.restartEngine(reconfigure: false) }
      }
    })
  }

  // MARK: Recognition

  private func startRequest() {
    guard let recognizer else { return }
    generation += 1
    let gen = generation
    task?.cancel()
    let req = SFSpeechAudioBufferRecognitionRequest()
    req.shouldReportPartialResults = true
    req.requiresOnDeviceRecognition = recognizer.supportsOnDeviceRecognition
    req.addsPunctuation = true
    req.taskHint = .dictation
    request = req
    box.set(req)
    lastText = ""
    task = recognizer.recognitionTask(with: req) { [weak self] result, error in
      // Extract plain values here: the result object is not Sendable.
      let text = result?.bestTranscription.formattedString
      let isFinal = result?.isFinal ?? false
      let failed = error != nil
      Task { @MainActor [weak self] in self?.handle(text: text, isFinal: isFinal, failed: failed, gen: gen) }
    }
  }

  private func handle(text: String?, isFinal: Bool, failed: Bool, gen: Int) {
    guard gen == generation, isListening else { return }
    if let text, !text.isEmpty, text != lastText {
      if lastText.isEmpty { lastUtteranceStart = Date() }
      lastText = text
      partial = text
      scheduleSilence(gen: gen)
    }
    if isFinal {
      finishUtterance(gen: gen)
    } else if failed {
      // The task ended on its own (recognizer time cap, no-speech timeout, audio glitch).
      // Keep whatever was heard, then start over after a short pause so a recognizer that
      // fails instantly (missing on-device model, revoked permission) cannot spin.
      Task { [weak self] in
        try? await Task.sleep(for: .milliseconds(500))
        self?.finishUtterance(gen: gen)
      }
    }
  }

  private func scheduleSilence(gen: Int) {
    silenceTimer?.cancel()
    let wait = silence
    silenceTimer = Task { [weak self] in
      try? await Task.sleep(for: .seconds(wait))
      guard !Task.isCancelled else { return }
      self?.finishUtterance(gen: gen)
    }
  }

  private func finishUtterance(gen: Int) {
    guard gen == generation else { return }
    silenceTimer?.cancel()
    let text = lastText.trimmingCharacters(in: .whitespacesAndNewlines)
    let finished = request
    startRequest()          // new request goes into the box first, so no audio lands on the old one
    finished?.endAudio()    // its final callback carries an old generation and is ignored
    partial = ""
    guard !text.isEmpty else { return }
    process(text)
  }

  /// Reset the recognizer context when playback ends, so Claude's own words are not carried
  /// into the wearer's next sentence.
  private func startEchoGuard() {
    echoTask?.cancel()
    echoTask = Task { [weak self] in
      var wasPlaying = false
      while !Task.isCancelled {
        try? await Task.sleep(for: .milliseconds(250))
        guard let self, self.isListening else { continue }
        let playing = self.isOutputPlaying()
        if wasPlaying, !playing { self.startRequest(); self.partial = "" }
        wasPlaying = playing
      }
    }
  }

  // MARK: Utterance policy

  private func process(_ raw: String) {
    utterances += 1
    lastUtterance = raw
    let lower = raw.lowercased()
    let short = lower.split(separator: " ").count <= 3
    if short, Self.stopWords.contains(where: { lower.contains($0) }) {
      onWake?()
      onUtterance?(raw)       // FrameRelay treats stop words locally
      return
    }
    let wake = wakeWord.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
    let wakeRange = (mode == .wake && !wake.isEmpty) ? lower.range(of: wake) : nil
    // The wake word barges in like a stop word. Anything else heard while the glasses are
    // talking is most likely the glasses themselves, so it is dropped.
    if wakeRange == nil, isOutputPlaying() {
      status = "listening (ignored: glasses talking)"
      return
    }
    switch mode {
    case .off:
      return
    case .always:
      emit(raw)
    case .wake:
      if let range = wakeRange {
        onWake?()
        // Indices come from the lowercased copy; reuse them on the original only when the
        // byte layout is identical (plain ASCII), otherwise fall back to the lowercased text.
        let source = raw.utf8.count == lower.utf8.count ? raw : lower
        let rest = String(source[range.upperBound...])
          .trimmingCharacters(in: CharacterSet.punctuationCharacters.union(.whitespacesAndNewlines))
        if rest.isEmpty {
          armedUntil = Date().addingTimeInterval(8)
          status = "armed: ask now"
        } else {
          emit(rest)
        }
      } else if Date() < armedUntil {
        emit(raw)
      } else {
        status = "listening (not for me)"
      }
    }
  }

  private func emit(_ text: String) {
    armedUntil = .distantPast
    status = "listening"
    onUtterance?(text)
  }
}
