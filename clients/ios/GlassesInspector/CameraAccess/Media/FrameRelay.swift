/*
 * FrameRelay.swift — Glasses Inspector addition.
 *
 * Pushes decoded preview frames from the glasses to the Mac receiver over a WebSocket.
 * Each message is 8 bytes of big-endian capture timestamp (ms since epoch) followed by
 * JPEG bytes.
 *
 * Transport is Network.framework (NWConnection + NWProtocolWebSocket) so the receiver can
 * be addressed by its Bonjour service name instead of an IP — link-local USB addresses
 * change every time the cable or Xcode cycles the interface — and so "prefer the USB
 * cable" is expressible as an interface constraint (Wi-Fi/cellular prohibited). If the
 * cable-only attempt cannot find a path, the next attempt allows any interface.
 *
 * Delivery: a single "latest frame" slot and one sender loop. A slow link lowers the
 * frame rate instead of building a backlog. Every send has a timeout.
 */

import Foundation
import Network
import QuartzCore
import Observation
import UIKit

enum RelayError: LocalizedError {
  case encodeFailed, noTarget, timeout, disconnected
  var errorDescription: String? {
    switch self {
    case .encodeFailed: return "JPEG encode failed"
    case .noTarget: return "no receiver selected"
    case .timeout: return "send timed out"
    case .disconnected: return "disconnected"
    }
  }
}

/// Network.framework objects are internally thread-safe; this lets the actor hand them
/// to callbacks without the compiler treating each capture as a data race.
private final class ConnectionBox: @unchecked Sendable {
  let c: NWConnection
  init(_ c: NWConnection) { self.c = c }
  var id: ObjectIdentifier { ObjectIdentifier(c) }
}

private enum StateSummary: Sendable {
  case ready, waiting(String), failed(String), cancelled, other
}

actor RelaySocket {
  enum Target: Equatable, Sendable {
    case url(URL)
    case service(name: String)
  }

  private var target: Target?
  /// Direct URL on the USB link (169.254.x.x) when the receiver advertised one.
  private var cableURL: URL?
  /// Direct URL on the LAN (advertised non-link-local address); avoids name resolution entirely.
  private var lanURL: URL?
  private var preferCable = true
  private var cableFailed = false
  private var cableFailedAt: Date?
  private var usingCable = false
  private var connection: ConnectionBox?
  private var ready = false
  private var readyWaiters: [CheckedContinuation<Void, Error>] = []
  private var lastFailure: String?
  private let queue = DispatchQueue(label: "glassesinspector.relay")

  /// Called with each text (JSON) message from the receiver.
  private var textHandler: (@Sendable (String) -> Void)?
  func setTextHandler(_ h: @escaping @Sendable (String) -> Void) { textHandler = h }

  /// Human-readable description of the path in use.
  private(set) var pathDescription: String = "not connected"
  /// Recent events for the in-app diagnostics panel.
  private(set) var events: [String] = []
  private func log(_ msg: String) {
    let f = DateFormatter(); f.dateFormat = "HH:mm:ss"
    events.append("\(f.string(from: Date())) \(msg)")
    if events.count > 40 { events.removeFirst(events.count - 40) }
  }

  func configure(target: Target, cableURL: URL?, lanURL: URL?, preferCable: Bool) {
    if self.target != target || self.preferCable != preferCable || self.cableURL != cableURL || self.lanURL != lanURL {
      close()
      cableFailed = false
      cableFailedAt = nil
    }
    self.target = target
    self.cableURL = cableURL
    self.lanURL = lanURL
    self.preferCable = preferCable
    log("target=\(target) cable=\(cableURL?.absoluteString ?? "none") lan=\(lanURL?.absoluteString ?? "none") preferCable=\(preferCable)")
  }

  /// All IPv4 interfaces on the phone (name, address), from the socket layer. The USB link
  /// is the one carrying a 169.254.x.x address; iOS keeps it off the default network path,
  /// so a plain NWPathMonitor never lists it.
  private func ipv4Interfaces() -> [(name: String, ip: String)] {
    var out: [(String, String)] = []
    var ifap: UnsafeMutablePointer<ifaddrs>?
    guard getifaddrs(&ifap) == 0, let first = ifap else { return [] }
    defer { freeifaddrs(ifap) }
    for ptr in sequence(first: first, next: { $0.pointee.ifa_next }) {
      let ifa = ptr.pointee
      guard let sa = ifa.ifa_addr, sa.pointee.sa_family == UInt8(AF_INET) else { continue }
      var a = sockaddr_in()
      memcpy(&a, sa, MemoryLayout<sockaddr_in>.size)
      out.append((String(cString: ifa.ifa_name), String(cString: inet_ntoa(a.sin_addr))))
    }
    return out
  }

  /// Name of the interface that carries the link-local (USB) address, if any.
  private func cableInterfaceName() -> String? {
    ipv4Interfaces().first { $0.ip.hasPrefix("169.254.") && $0.name != "lo0" }?.name
  }

  /// Ask typed path monitors for an NWInterface handle matching `name`.
  private func findCableInterface(named name: String, timeout: TimeInterval) async -> NWInterface? {
    final class Box: @unchecked Sendable { var found: NWInterface?; var seen: [String] = []; var done = false }
    for type in [NWInterface.InterfaceType.wiredEthernet, .other] {
      let box = Box()
      let monitor = NWPathMonitor(requiredInterfaceType: type)
      monitor.pathUpdateHandler = { path in
        box.seen = path.availableInterfaces.map { "\($0.name)/\($0.type)" }
        box.found = path.availableInterfaces.first { $0.name == name }
        box.done = true
      }
      monitor.start(queue: queue)
      let deadline = Date().addingTimeInterval(timeout)
      while !box.done && Date() < deadline { try? await Task.sleep(for: .milliseconds(25)) }
      monitor.cancel()
      log("monitor(\(type)): \(box.seen.isEmpty ? "none" : box.seen.joined(separator: ","))")
      if let f = box.found { return f }
    }
    return nil
  }

  func close() {
    connection?.c.cancel()
    connection = nil
    ready = false
    pathDescription = "not connected"
    let waiters = readyWaiters
    readyWaiters = []
    for w in waiters { w.resume(throwing: RelayError.disconnected) }
  }

  private func makeParameters(cableInterface: NWInterface?) -> NWParameters {
    let params = NWParameters.tcp
    let ws = NWProtocolWebSocket.Options()
    ws.autoReplyPing = true
    ws.maximumMessageSize = 8 * 1024 * 1024
    params.defaultProtocolStack.applicationProtocols.insert(ws, at: 0)
    params.includePeerToPeer = false
    if let cableInterface { params.requiredInterface = cableInterface }
    return params
  }

  private func endpoint(for target: Target) -> NWEndpoint {
    switch target {
    case .url(let url):
      return .url(url)
    case .service(let name):
      return .service(name: name, type: "_glassesrelay._tcp", domain: "local.", interface: nil)
    }
  }

  private func open() async throws {
    guard let target else { throw RelayError.noTarget }
    // Retry the cable a minute after it last failed (it may have been plugged in since).
    if cableFailed, let at = cableFailedAt, Date().timeIntervalSince(at) > 60 { cableFailed = false }
    var cableIface: NWInterface?
    var cableByProhibition = false
    // Any-interface attempt: prefer the advertised LAN IP (no resolver involved), then the service.
    var ep: NWEndpoint = lanURL.map { .url($0) } ?? endpoint(for: target)
    if preferCable && !cableFailed {
      let ifs = ipv4Interfaces()
      log("ifaces: " + ifs.map { "\($0.name)=\($0.ip)" }.joined(separator: " "))
      if let name = cableInterfaceName(), let cableURL {
        ep = .url(cableURL)
        if let iface = await findCableInterface(named: name, timeout: 0.6) {
          cableIface = iface
        } else {
          // No handle for it: forbid Wi-Fi/cellular and let the direct link-local address route.
          cableByProhibition = true
        }
      } else {
        cableFailed = true; cableFailedAt = Date()
        lastFailure = cableURL == nil ? "receiver advertised no 169.254 address" : "phone has no 169.254 interface"
        log("cable unavailable (\(lastFailure!)); using any interface")
      }
    }
    let cableOnly = cableIface != nil || cableByProhibition
    usingCable = cableOnly
    let params = makeParameters(cableInterface: cableIface)
    if cableByProhibition { params.prohibitedInterfaceTypes = [.wifi, .cellular] }
    log("open \(ep) via \(cableIface.map { "\($0.name)/\($0.type)" } ?? (cableByProhibition ? "non-Wi-Fi (169.254 route)" : "any interface"))")
    let box = ConnectionBox(NWConnection(to: ep, using: params))
    let id = box.id
    box.c.stateUpdateHandler = { [weak self] state in
      let summary: StateSummary
      switch state {
      case .ready: summary = .ready
      case .waiting(let e): summary = .waiting(e.localizedDescription)
      case .failed(let e): summary = .failed(e.localizedDescription)
      case .cancelled: summary = .cancelled
      default: summary = .other
      }
      Task { await self?.stateChanged(summary, id: id, cableAttempt: cableOnly) }
    }
    connection = box
    box.c.start(queue: queue)
    receiveNext(box)
  }

  private func stateChanged(_ state: StateSummary, id: ObjectIdentifier, cableAttempt: Bool) {
    guard let connection, connection.id == id else { return }
    log("state \(state)")
    switch state {
    case .ready:
      ready = true
      if let path = connection.c.currentPath, let iface = path.availableInterfaces.first {
        let kind: String
        switch iface.type {
        case .wifi: kind = "Wi-Fi"
        case .wiredEthernet: kind = "cable"
        case .cellular: kind = "cellular"
        default: kind = "usb/other"
        }
        pathDescription = "\(kind) \(iface.name)"
      } else {
        pathDescription = "connected"
      }
      let waiters = readyWaiters
      readyWaiters = []
      for w in waiters { w.resume() }
    case .waiting(let msg), .failed(let msg):
      // `.waiting` on a cable attempt means no usable cable path: fall back to any interface.
      if cableAttempt { cableFailed = true; cableFailedAt = Date() }
      lastFailure = msg
      close()
    case .cancelled:
      self.connection = nil
      ready = false
    case .other:
      break
    }
  }

  /// Drain incoming messages so pings and close frames are handled promptly.
  private func receiveNext(_ box: ConnectionBox) {
    box.c.receiveMessage { [weak self] content, context, _, error in
      if error == nil {
        if let content,
          let meta = context?.protocolMetadata(definition: NWProtocolWebSocket.definition) as? NWProtocolWebSocket.Metadata,
          meta.opcode == .text, let text = String(data: content, encoding: .utf8) {
          Task { await self?.deliverText(text) }
        }
        Task { await self?.receiveNext(box) }
      } else {
        Task { await self?.stateChanged(.failed(error?.localizedDescription ?? "receive failed"), id: box.id, cableAttempt: false) }
      }
    }
  }

  /// Waits for the connection to become ready. Polling (rather than a continuation) so a
  /// timeout or cancellation can never strand a task-group child.
  private func ensureReady(timeout: TimeInterval) async throws {
    if connection == nil {
      lastFailure = nil
      try await open()
    }
    // A cable attempt gets a short budget; a timeout there switches to any interface.
    let budget = usingCable ? min(timeout, 3) : timeout
    let deadline = Date().addingTimeInterval(budget)
    while !ready {
      if connection == nil {
        let detail = lastFailure.map { " (\($0))" } ?? ""
        throw NSError(domain: "Relay", code: 1, userInfo: [NSLocalizedDescriptionKey: "connect failed\(detail)"])
      }
      if Date() > deadline {
        if usingCable { cableFailed = true; cableFailedAt = Date() }
        let which = usingCable ? "USB" : "network"
        log("\(which) connect timed out after \(Int(budget))s")
        close()
        throw NSError(domain: "Relay", code: 2, userInfo: [NSLocalizedDescriptionKey: "\(which) connect timed out"])
      }
      try await Task.sleep(for: .milliseconds(40))
    }
  }

  private func deliverText(_ text: String) { textHandler?(text) }

  /// Sends a small JSON/text message to the receiver (commands such as "inspect").
  func sendText(_ text: String) async throws {
    try await ensureReady(timeout: 3)
    guard let box = connection else { throw RelayError.disconnected }
    try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
      let meta = NWProtocolWebSocket.Metadata(opcode: .text)
      let ctx = NWConnection.ContentContext(identifier: "cmd", metadata: [meta])
      box.c.send(content: Data(text.utf8), contentContext: ctx, isComplete: true, completion: .contentProcessed { err in
        if let err { cont.resume(throwing: err) } else { cont.resume() }
      })
    }
  }

  /// Encodes and sends one frame. Throws on timeout or socket failure; the socket is
  /// torn down on failure so the next call reconnects.
  func send(image: UIImage, quality: CGFloat, captureMillis: UInt64, timeout: TimeInterval) async throws -> Int {
    guard let data = image.jpegData(compressionQuality: quality) else { throw RelayError.encodeFailed }
    var payload = Data(capacity: data.count + 8)
    var ts = captureMillis.bigEndian
    payload.append(Data(bytes: &ts, count: 8))
    payload.append(data)
    try await ensureReady(timeout: timeout)
    guard let box = connection else { throw RelayError.disconnected }
    // Watchdog: cancelling the connection makes the pending send complete with an error,
    // so the continuation below always resumes.
    let watchdog = Task {
      try await Task.sleep(for: .seconds(timeout))
      box.c.cancel()
    }
    defer { watchdog.cancel() }
    do {
      try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
        let meta = NWProtocolWebSocket.Metadata(opcode: .binary)
        let ctx = NWConnection.ContentContext(identifier: "frame", metadata: [meta])
        box.c.send(content: payload, contentContext: ctx, isComplete: true, completion: .contentProcessed { err in
          if let err { cont.resume(throwing: err) } else { cont.resume() }
        })
      }
    } catch {
      close()
      throw watchdog.isCancelled ? error : RelayError.timeout
    }
    return payload.count
  }
}

/// Browses Bonjour for `_glassesrelay._tcp`. The receiver publishes its URLs in the TXT
/// record, which the settings sheet shows for reference; connections use the service name.
@Observable
@MainActor
final class RelayBrowser {
  struct Receiver: Identifiable, Equatable {
    var id: String { name }
    let name: String
    let urls: [String]
  }
  private(set) var found: [Receiver] = []
  var events: [String] = []
  var onUpdate: (() -> Void)?
  private var browser: NWBrowser?

  func start() {
    guard browser == nil else { return }
    let params = NWParameters()
    params.includePeerToPeer = false
    let b = NWBrowser(for: .bonjourWithTXTRecord(type: "_glassesrelay._tcp", domain: "local."), using: params)
    b.browseResultsChangedHandler = { [weak self] results, _ in
      var list: [Receiver] = []
      for r in results {
        guard case .service(let name, _, _, _) = r.endpoint else { continue }
        var urls: [String] = []
        if case .bonjour(let txt) = r.metadata, let u = txt.dictionary["urls"] {
          urls = u.split(separator: ",").map(String.init)
        }
        if !list.contains(where: { $0.name == name }) { list.append(Receiver(name: name, urls: urls)) }
      }
      Task { @MainActor [weak self] in
        guard let self else { return }
        if self.found != list {
          self.found = list
          self.events.append("browse: \(list.map { $0.name }.joined(separator: ", ").ifEmpty("nothing"))")
          self.onUpdate?()
        }
      }
    }
    b.stateUpdateHandler = { [weak self] state in
      Task { @MainActor [weak self] in
        guard let self else { return }
        self.events.append("browser \(state)")
        if case .failed = state {
          self.stop()
          try? await Task.sleep(for: .seconds(1))
          self.start()
        }
      }
    }
    b.start(queue: .main)
    browser = b
  }

  func stop() {
    browser?.cancel()
    browser = nil
  }
}

/// Per-frame path, entirely off the main thread: throttle, latest-frame slot, encode, send, counters.
actor RelayEngine {
  struct Snapshot: Sendable {
    var inputFPS = 0.0, outputFPS = 0.0
    var sent = 0, dropped = 0, failed = 0, inFlight = 0
    var frameKB = 0, encodeMillis = 0
    var error: String?
    var path = "not connected"
    var events: [String] = []
  }

  let socket = RelaySocket()
  var enabled = true
  var targetFPS = 24.0
  var quality: CGFloat = 0.7
  var maxInFlight = 3

  private var latest: (image: UIImage, millis: UInt64)?
  private var inFlight = 0
  private var lastAccepted: TimeInterval = 0
  private var inputTimes: [TimeInterval] = []
  private var sendTimes: [TimeInterval] = []
  private var sent = 0, dropped = 0, failed = 0, frameKB = 0, encodeMillis = 0
  private var error: String?

  func configure(enabled: Bool, targetFPS: Double, quality: Double, maxInFlight: Int) {
    self.enabled = enabled; self.targetFPS = targetFPS; self.quality = quality; self.maxInFlight = maxInFlight
  }

  func ingest(_ image: UIImage, at now: TimeInterval) {
    guard enabled else { return }
    inputTimes.append(now)
    if inputTimes.count > 64 { inputTimes.removeFirst(inputTimes.count - 64) }
    // Accept a little early so jitter at input ≈ cap doesn't reject every third frame.
    guard now - lastAccepted >= 0.8 / max(targetFPS, 1) else { return }
    lastAccepted = now
    if inFlight < maxInFlight {
      startSend(image, millis: UInt64(now * 1000))
    } else {
      if latest != nil { dropped += 1 }
      latest = (image, UInt64(now * 1000))
    }
  }

  private func startSend(_ image: UIImage, millis: UInt64) {
    inFlight += 1
    Task {
      let t0 = Date()
      do {
        let bytes = try await socket.send(image: image, quality: quality, captureMillis: millis, timeout: 4)
        sent += 1
        frameKB = bytes / 1024
        error = nil
        let now = Date().timeIntervalSince1970
        sendTimes.append(now)
        if sendTimes.count > 64 { sendTimes.removeFirst(sendTimes.count - 64) }
      } catch {
        failed += 1
        self.error = error.localizedDescription
        try? await Task.sleep(for: .milliseconds(500))
      }
      encodeMillis = Int(Date().timeIntervalSince(t0) * 1000)
      inFlight -= 1
      if let next = latest {
        latest = nil
        startSend(next.image, millis: next.millis)
      }
    }
  }

  private static func rate(_ times: [TimeInterval], now: TimeInterval) -> Double {
    let recent = times.filter { now - $0 <= 2 }
    guard let first = recent.first, recent.count > 1, now > first else { return 0 }
    return Double(recent.count - 1) / (now - first)
  }

  func snapshot() async -> Snapshot {
    let now = Date().timeIntervalSince1970
    return Snapshot(inputFPS: Self.rate(inputTimes, now: now), outputFPS: Self.rate(sendTimes, now: now),
                    sent: sent, dropped: dropped, failed: failed, inFlight: inFlight,
                    frameKB: frameKB, encodeMillis: encodeMillis, error: error,
                    path: await socket.pathDescription, events: await socket.events)
  }
}

/// Main-thread health: how late a 100 ms timer fires (stall) and the process CPU load.
@MainActor
final class MainThreadMeter {
  private var timer: Timer?
  private var expected: TimeInterval = 0
  private var maxLate: TimeInterval = 0
  private(set) var stallMillis = 0
  private var lastCPU: (user: Double, sys: Double, at: TimeInterval)?
  private(set) var cpuPercent = 0

  func start() {
    guard timer == nil else { return }
    expected = CACurrentMediaTime() + 0.1
    timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
      Task { @MainActor [weak self] in self?.tick() }
    }
  }

  private func tick() {
    let now = CACurrentMediaTime()
    maxLate = max(maxLate, now - expected)
    expected = now + 0.1
  }

  /// Publishes the worst stall since the last call and samples CPU.
  func sample() {
    stallMillis = Int(maxLate * 1000)
    maxLate = 0
    var info = task_thread_times_info()
    var count = mach_msg_type_number_t(MemoryLayout<task_thread_times_info>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
      $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
        task_info(mach_task_self_, task_flavor_t(TASK_THREAD_TIMES_INFO), $0, &count)
      }
    }
    guard kr == KERN_SUCCESS else { return }
    let user = Double(info.user_time.seconds) + Double(info.user_time.microseconds) / 1e6
    let sys = Double(info.system_time.seconds) + Double(info.system_time.microseconds) / 1e6
    let at = Date().timeIntervalSince1970
    if let last = lastCPU, at > last.at {
      cpuPercent = Int(((user - last.user) + (sys - last.sys)) / (at - last.at) * 100)
    }
    lastCPU = (user, sys, at)
  }
}

@Observable
@MainActor
final class FrameRelay {
  static let serviceKey = "relayService"
  static let manualURLKey = "relayManualURL"
  static let useManualKey = "relayUseManual"
  static let preferCableKey = "relayPreferCable"
  static let fpsKey = "relayFPS"
  static let qualityKey = "relayQuality"

  /// Bonjour name of the chosen receiver ("" = first one discovered).
  var serviceName: String {
    didSet { UserDefaults.standard.set(serviceName, forKey: Self.serviceKey); applyTarget() }
  }
  /// Manual URL, used only when `useManualURL` is on.
  var manualURL: String {
    didSet { UserDefaults.standard.set(manualURL, forKey: Self.manualURLKey); applyTarget() }
  }
  var useManualURL: Bool {
    didSet { UserDefaults.standard.set(useManualURL, forKey: Self.useManualKey); applyTarget() }
  }
  /// Use the USB cable when one is present.
  var preferCable: Bool {
    didSet { UserDefaults.standard.set(preferCable, forKey: Self.preferCableKey); applyTarget() }
  }
  var isEnabled: Bool = true { didSet { pushSettings() } }
  var targetFPS: Double {
    didSet { UserDefaults.standard.set(targetFPS, forKey: Self.fpsKey); pushSettings() }
  }
  var jpegQuality: Double {
    didSet { UserDefaults.standard.set(jpegQuality, forKey: Self.qualityKey); pushSettings() }
  }
  var maxInFlight: Int = 3 { didSet { pushSettings() } }

  // MARK: Published snapshot (refreshed twice a second; the camera screen never
  // re-renders per frame because of the relay).
  private(set) var statusText: String = "idle"
  private(set) var pathDescription: String = "not connected"
  private(set) var socketEvents: [String] = []
  private(set) var sentFrames: Int = 0
  private(set) var droppedFrames: Int = 0
  private(set) var failedFrames: Int = 0
  private(set) var lastError: String?
  private(set) var lastFrameKB: Int = 0
  private(set) var effectiveFPS: Double = 0
  private(set) var inputFPS: Double = 0
  private(set) var uiStallMillis: Int = 0
  private(set) var cpuPercent: Int = 0
  private(set) var encodeMillis: Int = 0

  var diagnostics: [String] { (browser.events.suffix(4) + socketEvents.suffix(14)) }

  var activeTargetDescription: String {
    if useManualURL { return "\(manualURL) via \(pathDescription)" }
    if let name = serviceName.isEmpty ? browser.found.first?.name : serviceName {
      return "\(name) via \(pathDescription)"
    }
    return "nothing discovered, using \(manualURL) via \(pathDescription)"
  }

  let browser = RelayBrowser()
  private let engine = RelayEngine()
  private let meter = MainThreadMeter()
  @ObservationIgnored private var statsTask: Task<Void, Never>?

  // MARK: Claude analysis -> speech
  let speaker = Speaker()
  /// Speak analysis sentences from the Mac through the glasses.
  var speakEnabled: Bool {
    didSet { UserDefaults.standard.set(speakEnabled, forKey: "relaySpeak") }
  }
  /// Live text of the current/last analysis (for the on-screen caption).
  private(set) var caption: String = ""
  private(set) var captionFinal: Bool = true
  /// Server-side continuous narration state, as reported by the receiver.
  private(set) var narrationEnabled: Bool = false
  private(set) var narrationInterval: Double = 8
  private(set) var lastCommandError: String?

  init() {
    let d = UserDefaults.standard
    serviceName = d.string(forKey: Self.serviceKey) ?? ""
    manualURL = d.string(forKey: Self.manualURLKey) ?? d.string(forKey: "relayURL") ?? "http://Eddies-MacBook-Pro.local:8787"
    useManualURL = d.bool(forKey: Self.useManualKey)
    preferCable = d.object(forKey: Self.preferCableKey) as? Bool ?? true
    targetFPS = d.object(forKey: Self.fpsKey) as? Double ?? 24
    jpegQuality = d.object(forKey: Self.qualityKey) as? Double ?? 0.7
    speakEnabled = d.object(forKey: "relaySpeak") as? Bool ?? true
    browser.onUpdate = { [weak self] in self?.applyTarget() }
    browser.start()
    meter.start()
    applyTarget()
    pushSettings()
    Task { [engine] in
      await engine.socket.setTextHandler { [weak self] text in
        Task { @MainActor [weak self] in self?.handleServerText(text) }
      }
    }
    statsTask = Task { [weak self] in
      while !Task.isCancelled {
        try? await Task.sleep(for: .milliseconds(500))
        await self?.refreshStats()
      }
    }
  }

  private func pushSettings() {
    let e = isEnabled, f = targetFPS, q = jpegQuality, m = maxInFlight
    Task { [engine] in await engine.configure(enabled: e, targetFPS: f, quality: q, maxInFlight: m) }
  }

  private func refreshStats() async {
    let snap = await engine.snapshot()
    meter.sample()
    inputFPS = snap.inputFPS
    effectiveFPS = snap.outputFPS
    sentFrames = snap.sent
    droppedFrames = snap.dropped
    failedFrames = snap.failed
    lastError = snap.error
    lastFrameKB = snap.frameKB
    encodeMillis = snap.encodeMillis
    uiStallMillis = meter.stallMillis
    cpuPercent = meter.cpuPercent
    if snap.path != pathDescription { pathDescription = snap.path }
    if snap.events != socketEvents { socketEvents = snap.events }
    statusText = snap.error ?? "in \(String(format: "%.1f", inputFPS)) · out \(String(format: "%.1f", effectiveFPS)) fps · \(lastFrameKB) KB · enc \(encodeMillis) ms · ui stall \(uiStallMillis) ms · cpu \(cpuPercent)% · \(sentFrames) sent"
  }

  private func applyTarget() {
    let target: RelaySocket.Target?
    if useManualURL {
      if var comps = URLComponents(string: manualURL.trimmingCharacters(in: .whitespaces)) {
        let secure = comps.scheme == "https" || comps.scheme == "wss"
        comps.scheme = secure ? "wss" : "ws"
        comps.path = "/ws/ingest"
        target = comps.url.map { .url($0) }
      } else {
        target = nil
      }
    } else if let name = serviceName.isEmpty ? browser.found.first?.name : serviceName {
      target = .service(name: name)
    } else if var comps = URLComponents(string: manualURL.trimmingCharacters(in: .whitespaces)) {
      comps.scheme = "ws"
      comps.path = "/ws/ingest"
      target = comps.url.map { .url($0) }
    } else {
      target = nil
    }
    guard let target else { return }
    func wsURL(_ u: String) -> URL? {
      guard var comps = URLComponents(string: u) else { return nil }
      comps.scheme = "ws"
      comps.path = "/ws/ingest"
      return comps.url
    }
    var cableURL: URL?
    var lanURL: URL?
    if case .service(let name) = target, let r = browser.found.first(where: { $0.name == name }) {
      cableURL = r.urls.first(where: { $0.contains("169.254") }).flatMap(wsURL)
      lanURL = r.urls.first(where: { !$0.contains("169.254") }).flatMap(wsURL)
    }
    let prefer = preferCable
    Task { [engine] in await engine.socket.configure(target: target, cableURL: cableURL, lanURL: lanURL, preferCable: prefer) }
  }

  /// Called from the toolkit's frame callback (any thread). Never touches the main actor.
  nonisolated func push(_ image: UIImage) {
    let now = Date().timeIntervalSince1970
    Task { [engine] in await engine.ingest(image, at: now) }
  }

  /// Ask the Mac to analyze the current frame; the answer comes back as speech.
  func requestInspect(question: String? = nil) {
    caption = ""
    captionFinal = false
    var msg: [String: Any] = ["type": "inspect"]
    if let question, !question.isEmpty { msg["question"] = question }
    sendCommand(msg)
  }

  func setNarration(enabled: Bool, interval: Double) {
    sendCommand(["type": "narrate", "enabled": enabled, "interval": interval])
  }

  private func sendCommand(_ msg: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: msg), let text = String(data: data, encoding: .utf8) else { return }
    Task { [engine] in
      do { try await engine.socket.sendText(text); await MainActor.run { self.lastCommandError = nil } }
      catch { await MainActor.run { self.lastCommandError = error.localizedDescription } }
    }
  }

  private func handleServerText(_ text: String) {
    guard let data = text.data(using: .utf8),
      let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
      let kind = obj["type"] as? String else { return }
    switch kind {
    case "speak":
      if let t = obj["text"] as? String {
        caption = caption.isEmpty ? t : caption + " " + t
        captionFinal = false
        if speakEnabled { speaker.speak(t) }
      }
    case "speak_end":
      captionFinal = true
    case "narration":
      narrationEnabled = obj["enabled"] as? Bool ?? false
      narrationInterval = obj["interval"] as? Double ?? narrationInterval
    default:
      break
    }
  }
}

private extension String {
  func ifEmpty(_ alt: String) -> String { isEmpty ? alt : self }
}
