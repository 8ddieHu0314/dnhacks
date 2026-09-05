/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

//
// WearablesViewModel.swift
//
// Primary view model for the CameraAccess app that manages DAT SDK integration.
// Demonstrates how to listen to device availability changes using the DAT SDK's
// device stream functionality and handle permission requests.
//

import MWDATCore
import Observation
import SwiftUI

@Observable
@MainActor
final class WearablesViewModel {
  var devices: [DeviceIdentifier]
  var registrationState: RegistrationState
  var showError: Bool = false
  var errorMessage: String = ""
  var requiresFirmwareUpdate: Bool = false

  @ObservationIgnored private var registrationTask: Task<Void, Never>?
  @ObservationIgnored private var deviceStreamTask: Task<Void, Never>?
  @ObservationIgnored private var setupDeviceStreamTask: Task<Void, Never>?
  private let wearables: WearablesInterface
  private var deviceCompatibility: [DeviceIdentifier: Compatibility] = [:]
  private var compatibilityListenerTokens: [DeviceIdentifier: AnyListenerToken] = [:]

  init(wearables: WearablesInterface) {
    self.wearables = wearables
    self.devices = wearables.devices
    self.registrationState = wearables.registrationState

    // Set up device stream immediately to handle MockDevice events
    setupDeviceStreamTask = Task {
      await setupDeviceStream()
    }

    registrationTask = Task {
      for await registrationState in wearables.registrationStateStream() {
        self.registrationState = registrationState
      }
    }
  }

  isolated deinit {
    registrationTask?.cancel()
    deviceStreamTask?.cancel()
    setupDeviceStreamTask?.cancel()
  }

  private func setupDeviceStream() async {
    if let task = deviceStreamTask, !task.isCancelled {
      task.cancel()
    }

    deviceStreamTask = Task {
      for await devices in wearables.devicesStream() {
        self.devices = devices
        // Monitor compatibility for each device
        monitorDeviceCompatibility(devices: devices)
      }
    }
  }

  // MARK: - Glasses Inspector: device link diagnostics

  /// One line per known device: name, type, link state, compatibility. Refreshed on
  /// every device-list change and every link-state change.
  var deviceStatusText: String = "no devices"
  @ObservationIgnored private var linkListenerTokens: [DeviceIdentifier: AnyListenerToken] = [:]

  private func refreshDeviceStatus() {
    let lines = devices.compactMap { id -> String? in
      guard let d = wearables.deviceForIdentifier(id) else { return nil }
      return "\(d.nameOrId()) [\(d.deviceType().rawValue)] link=\(d.linkState) compat=\(d.compatibility().displayString)"
    }
    deviceStatusText = lines.isEmpty ? "no devices" : lines.joined(separator: "\n")
  }

  private func monitorLinkState(devices: [DeviceIdentifier]) {
    let deviceSet = Set(devices)
    linkListenerTokens = linkListenerTokens.filter { deviceSet.contains($0.key) }
    for id in devices where linkListenerTokens[id] == nil {
      guard let d = wearables.deviceForIdentifier(id) else { continue }
      linkListenerTokens[id] = d.addLinkStateListener { [weak self] _ in
        Task { @MainActor [weak self] in self?.refreshDeviceStatus() }
      }
    }
    refreshDeviceStatus()
  }

  private func monitorDeviceCompatibility(devices: [DeviceIdentifier]) {
    monitorLinkState(devices: devices)
    // Remove listeners for devices that are no longer present
    let deviceSet = Set(devices)
    compatibilityListenerTokens = compatibilityListenerTokens.filter { deviceSet.contains($0.key) }
    deviceCompatibility = deviceCompatibility.filter { deviceSet.contains($0.key) }
    updateFirmwareUpdateRequired()

    // Add listeners for new devices
    for deviceId in devices {
      guard compatibilityListenerTokens[deviceId] == nil else { continue }
      guard let device = wearables.deviceForIdentifier(deviceId) else { continue }
      deviceCompatibility[deviceId] = device.compatibility()
      updateFirmwareUpdateRequired()

      // Capture device name before the closure to avoid Sendable issues
      let deviceName = device.nameOrId()
      let token = device.addCompatibilityListener { [weak self] compatibility in
        Task { [weak self] in
          await self?.handleCompatibilityChange(
            compatibility,
            deviceId: deviceId,
            deviceName: deviceName
          )
        }
      }
      compatibilityListenerTokens[deviceId] = token
    }
  }

  func connectGlasses() {
    guard registrationState != .registering else { return }
    Task { @MainActor in
      do {
        try await wearables.startRegistration()
      } catch let error as RegistrationError {
        showError(error.description)
      } catch {
        showError(error.localizedDescription)
      }
    }
  }

  func disconnectGlasses() {
    Task { @MainActor in
      do {
        try await wearables.startUnregistration()
      } catch let error as UnregistrationError {
        showError(error.description)
      } catch {
        showError(error.localizedDescription)
      }
    }
  }

  func openFirmwareUpdate() async {
    do {
      try await wearables.openFirmwareUpdate()
    } catch {
      showError(error.description)
    }
  }

  func openDATGlassesAppUpdate() async {
    do {
      try await wearables.openDATGlassesAppUpdate()
    } catch {
      showError(error.description)
    }
  }

  func showError(_ error: String) {
    errorMessage = error
    showError = true
  }

  func dismissError() {
    showError = false
  }

  private func updateFirmwareUpdateRequired() {
    requiresFirmwareUpdate = deviceCompatibility.values.contains(.deviceUpdateRequired)
  }

  private func handleCompatibilityChange(
    _ compatibility: Compatibility,
    deviceId: DeviceIdentifier,
    deviceName: String
  ) {
    deviceCompatibility[deviceId] = compatibility
    updateFirmwareUpdateRequired()
    if compatibility == .deviceUpdateRequired {
      showError("\(deviceName) needs an update to work with this app.")
    }
  }
}
