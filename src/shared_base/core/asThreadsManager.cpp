/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS HEADER.
 *
 * The contents of this file are subject to the terms of the
 * Common Development and Distribution License (the "License").
 * You may not use this file except in compliance with the License.
 *
 * You can read the License at http://opensource.org/licenses/CDDL-1.0
 * See the License for the specific language governing permissions
 * and limitations under the License.
 *
 * When distributing Covered Code, include this CDDL Header Notice in
 * each file and include the License file (licence.txt). If applicable,
 * add the following below this CDDL Header, with the fields enclosed
 * by brackets [] replaced by your own identifying information:
 * "Portions Copyright [year] [name of copyright owner]"
 *
 * The Original Software is AtmoSwing.
 * The Original Software was developed at the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#include "asThreadsManager.h"

#include <wx/fileconf.h>

#include "asIncludes.h"
#include "asThread.h"

// Safe: Critical section defined within
asThreadsManager::asThreadsManager()
    : _idCounter(-1),
      _pendingTotal(0),
      _waitingUntilAllDone(true),
      _cancelled(false),
      _maxThreadsNb(-1),
      _priority(-1) {}

int asThreadsManager::GetPendingNbLocked(int type) const {
    if (type < 0) {
        return _pendingTotal;
    }

    auto it = _pendingByType.find(type);

    return it == _pendingByType.end() ? 0 : it->second;
}

void asThreadsManager::ReleasePending(int type) {
    {
        std::lock_guard<std::mutex> lock(_pendingMutex);
        auto it = _pendingByType.find(type);
        if (it != _pendingByType.end() && it->second > 0) {
            --it->second;
            --_pendingTotal;
        } else {
            wxLogError(_("Thread accounting underflow for type %d."), type);
        }
    }

    _pendingCondition.notify_all();
}

asThreadsManager::~asThreadsManager() {
    // Destroying the manager while workers are still running would leave them calling into
    // freed members from OnExit(). Callers are expected to have waited already; flag it
    // rather than crash obscurely later.
    if (GetRunningThreadsNb() > 0) {
        wxLogError(_("The threads manager is being destroyed while %d thread(s) are still running."),
                   GetRunningThreadsNb());
    }
}

void asThreadsManager::Init() {
    _critSectionManager.Enter();

    wxConfigBase* pConfig = wxFileConfig::Get();
    _maxThreadsNb = wxThread::GetCPUCount();
    if (_maxThreadsNb == -1) _maxThreadsNb = 2;
    pConfig->Read("/Processing/ThreadsNb", &_maxThreadsNb, _maxThreadsNb);
    _priority = pConfig->Read("/Processing/ThreadsPriority", 95l);

    _critSectionManager.Leave();
}

void asThreadsManager::OnClose(wxCloseEvent&) {
    if (GetRunningThreadsNb() > 0) {
        CleanArray();
    }
}

// Safe: Critical section defined within
int asThreadsManager::GetTotalThreadsNb() {
    wxCriticalSectionLocker lock(_critSectionManager);

    return (int)_threads.size();
}

// Number of threads whose Entry() has not returned yet. Counted from our own registrations
// rather than from wxThread::IsRunning(), which turns false too early (see _pendingMutex).
int asThreadsManager::GetRunningThreadsNb(int type) {
    std::lock_guard<std::mutex> lock(_pendingMutex);

    return GetPendingNbLocked(type);
}

// Safe: Critical section defined within
int asThreadsManager::GetFreeDevice(int devicesNb) {
    if (devicesNb == 1) {
        return 0;
    }

    _critSectionManager.Enter();

    int deviceOccupancy = 0;
    int selectedDevice = 0;

    for (int device = 0; device < devicesNb; ++device) {
        int counter = 0;
        for (auto& thread : _threads) {
            if (thread->GetDevice() == device) {
                counter++;
            }
        }
        if (counter == 0) {
            _critSectionManager.Leave();
            return device;
        }
        if (device == 0) {
            deviceOccupancy = counter;
        } else if (counter < deviceOccupancy) {
            selectedDevice = device;
        }
    }

    _critSectionManager.Leave();

    return selectedDevice;
}

// Safe: Critical section defined in GetRunningThreadsNb
int asThreadsManager::GetAvailableThreadsNb() {
    if (_maxThreadsNb < 1) Init();

    // Maximum threads nb
    int runningThreads = GetRunningThreadsNb();
    _critSectionManager.Enter();
    int nb = _maxThreadsNb - runningThreads;
    _critSectionManager.Leave();

    if (nb < 1) {
        nb = 1;
    }

    return nb;
}

// Safe: Critical section defined within
bool asThreadsManager::AddThread(asThread* thread) {
    const int type = thread->GetType();

    // Register the thread as pending *before* it can possibly start, so that a Wait() issued
    // immediately after this call cannot observe zero and return early.
    {
        std::lock_guard<std::mutex> lock(_pendingMutex);
        ++_pendingByType[type];
        ++_pendingTotal;
    }

    // Create. Failure here means the thread was never spawned — direct `delete` is safe
    // (no race with thread teardown; the thread is not yet in `_threads`).
    if (thread->Create() != wxTHREAD_NO_ERROR) {
        wxLogError(_("Cannot create the thread !"));
        ReleasePending(type);
        delete thread;
        return false;
    }

    // Check the number of threads currently running
    if (GetAvailableThreadsNb() < 1) {
        wxLogWarning(_("The thread array is currently full."));
    }

    // Set priority
    if (_priority < 0) Init();
    thread->SetPriority(_priority);

    // Add to array before Run(): once running, the thread may finish and call RemoveThread()
    // at any moment, and RemoveThread() has to find it there.
    _critSectionManager.Enter();
    _threads.push_back(thread);
    _critSectionManager.Leave();

    // Run. If Run() fails the OS-level thread never started, so OnExit() will not run and we
    // have to undo both the array entry and the pending registration ourselves.
    if (thread->Run() != wxTHREAD_NO_ERROR) {
        wxLogError(_("Can't run the thread!"));
        _critSectionManager.Enter();
        for (auto it = _threads.begin(); it != _threads.end(); ++it) {
            if (*it == thread) {
                _threads.erase(it);
                break;
            }
        }
        _critSectionManager.Leave();
        ReleasePending(type);
        delete thread;
        return false;
    }

    _critSectionManager.Enter();
    _cancelled = false;
    _waitingUntilAllDone = true;
    _critSectionManager.Leave();

    return true;
}

void asThreadsManager::RemoveThread(asThread* thread) {
    const int type = thread->GetType();
    bool found = false;

    // Drop the entry before releasing the pending count: once the count reaches zero a
    // waiter may return, and by then the manager must no longer hold a pointer to a thread
    // object that wxWidgets is about to delete.
    _critSectionManager.Enter();
    for (auto it = _threads.begin(); it != _threads.end(); ++it) {
        if (*it == thread) {
            _threads.erase(it);
            found = true;
            break;
        }
    }
    _critSectionManager.Leave();

    if (!found) {
        wxLogError(_("Thread %p couldn't be removed."), static_cast<void*>(thread));
    }

    ReleasePending(type);
}

// Safe: Critical section defined within
bool asThreadsManager::CleanArray() {
    _critSectionManager.Enter();

    // RemoveThread() erases each entry as its thread finishes, so the array only ever holds
    // live threads. Anything left here is still running and must not be touched.
    bool empty = _threads.empty();
    if (empty) {
        _idCounter = 0;
    }

    _critSectionManager.Leave();

    return empty;
}

void asThreadsManager::Wait(int type) {
    std::unique_lock<std::mutex> lock(_pendingMutex);
    _pendingCondition.wait(lock, [this, type] { return GetPendingNbLocked(type) == 0; });
}

bool asThreadsManager::HasFreeThread(int type) {
    if (_maxThreadsNb < 1) Init();

    return _maxThreadsNb - GetRunningThreadsNb(type) > 0;
}

void asThreadsManager::WaitForFreeThread(int type) {
    if (_maxThreadsNb < 1) Init();

    // Snapshot under its own critical section: _maxThreadsNb is written by Init().
    _critSectionManager.Enter();
    const int maxThreadsNb = _maxThreadsNb;
    _critSectionManager.Leave();

    std::unique_lock<std::mutex> lock(_pendingMutex);
    _pendingCondition.wait(lock, [this, type, maxThreadsNb] { return maxThreadsNb - GetPendingNbLocked(type) > 0; });
}

void asThreadsManager::PauseAll() {
    wxCriticalSectionLocker lock(_critSectionManager);

    for (auto& thread : _threads) {
        if (thread->IsRunning()) {
            //            thread->Pause();
        }
    }
}

void asThreadsManager::ResumeAll() {
    wxCriticalSectionLocker lock(_critSectionManager);

    for (auto& thread : _threads) {
        if (thread->IsPaused()) {
            thread->Resume();
        }
    }
}
